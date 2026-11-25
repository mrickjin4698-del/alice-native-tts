"""Dispatch the single-request CUDA RF schedule without step-by-step host reads."""


def batched_tail_scan(torch, original):
    def find_flattening_point(latent, target_value=0.0, window_size=20, std_threshold=0.05, mean_threshold=0.1):
        if latent.ndim != 2 or not latent.is_cuda or not latent.shape[0] or window_size <= 0:
            return original(latent, target_value, window_size, std_threshold, mean_threshold)
        count, channels = latent.shape
        padded = torch.cat((latent, latent.new_zeros((window_size, channels))), dim=0)
        windows = padded.unfold(0, window_size, 1)[:count].transpose(1, 2)
        std = windows.std(dim=(1, 2), unbiased=False)
        mean_distance = (windows.mean(dim=(1, 2)) - target_value).abs()
        valid = (std < std_threshold) & (mean_distance < mean_threshold)
        # A different reduction layout can round across a threshold. Preserve
        # the upstream decision when such a window could change the first match.
        epsilon = 8 * torch.finfo(latent.dtype).eps
        near = ((std - std_threshold).abs() <= epsilon) | ((mean_distance - mean_threshold).abs() <= epsilon)
        indices = torch.arange(count, device=latent.device)
        first, uncertain = torch.stack((
            torch.where(valid, indices, count).amin(),
            torch.where(near, indices, count).amin(),
        )).tolist()
        if uncertain <= first and uncertain < count:
            return original(latent, target_value, window_size, std_threshold, mean_threshold)
        return first

    return find_flattening_point


def linear_rf_sampler(torch, original, draw_initial_noise, temporal_score_rescale):
    condition_names = (
        "text_input_ids", "text_mask", "ref_latent", "ref_mask", "caption_input_ids",
        "caption_mask", "speaker_state_override", "speaker_mask_override", "speaker_uncond_mode",
    )
    output_names = ("text_state", "text_mask", "speaker_state", "speaker_mask", "caption_state", "caption_mask")
    last_schedule = None

    def with_cache(model, states):
        kwargs = dict(zip(output_names, states, strict=True))
        kwargs["context_kv_cache"] = model.build_context_kv_cache(
            text_state=states[0], speaker_state=states[2], caption_state=states[4]
        )
        return kwargs

    def sample(model, **options):
        nonlocal last_schedule
        if (
            options["cfg_guidance_mode"] != "independent"
            or options["t_schedule_mode"] != "linear"
            or options["speaker_uncond_mode"] != "mask"
            or not options["use_context_kv_cache"]
            or "seed" not in options
            or any(options.get(name) is not None for name in (
                "initial_noise", "latent_mask", "speaker_kv_scale", "cfg_scale",
            ))
        ):
            return original(model, **options)

        count = int(options["num_steps"])
        text_ids = options["text_input_ids"]
        device, dtype = model.device, model.dtype
        batch = text_ids.shape[0]
        latent, _, _ = draw_initial_noise(
            options["seed"], (batch, options["sequence_length"], model.cfg.patched_latent_dim),
            device=device, dtype=dtype,
        )
        if options["truncation_factor"] is not None:
            latent = latent * float(options["truncation_factor"])
        conditions = model.encode_conditions(**{name: options[name] for name in condition_names})
        enabled = (
            options["cfg_scale_text"] > 0,
            model.cfg.use_speaker_condition_resolved and options["cfg_scale_speaker"] > 0,
            model.cfg.use_caption_condition and options["cfg_scale_caption"] > 0
            and conditions[5] is not None and bool(conditions[5].any().item()),
        )
        cfg_bundles = [conditions]
        strengths = []
        for part, applies in enumerate(enabled):
            if applies:
                unguided = list(conditions)
                offset = 2 * part
                unguided[offset] = torch.zeros_like(conditions[offset])
                unguided[offset + 1] = torch.zeros_like(conditions[offset + 1])
                cfg_bundles.append(tuple(unguided))
                name = ("cfg_scale_text", "cfg_scale_speaker", "cfg_scale_caption")[part]
                strengths.append(float(options[name]))

        conditional = with_cache(model, conditions)
        branches = len(cfg_bundles)
        guidance = None
        if branches > 1:
            rows = tuple(torch.cat(items, dim=0) if items[0] is not None else None for items in zip(*cfg_bundles))
            guidance = with_cache(model, rows)

        width = batch * branches
        schedule_key = (device, dtype, count, width)
        if last_schedule is None or last_schedule[0] != schedule_key:
            schedule = (1.0 - torch.linspace(0.0, 1.0, count + 1, device=device)) * 0.999
            times = schedule.tolist()
            if any(first <= second for first, second in zip(times, times[1:])):
                raise ValueError("t_schedule must be strictly decreasing; adjust num_steps")
            steps = schedule[1:] - schedule[:-1]
            timestep_table = schedule[:-1].to(dtype)[:, None]
            # Give every slice of the timestep table a 256-byte-aligned address.
            boundary = 256 // timestep_table.element_size()
            columns = ((width + boundary - 1) // boundary) * boundary
            last_schedule = (schedule_key, schedule, times, steps, timestep_table.repeat(1, columns))
        _, schedule, times, steps, time_inputs = last_schedule

        lower, upper = options["cfg_min_t"], options["cfg_max_t"]
        callback = options.get("step_callback")
        for step in range(count):
            if callback is not None:
                callback(step, count)
            if guidance is not None and lower <= times[step] <= upper:
                output = model.forward_with_encoded_conditions(
                    x_t=torch.cat([latent] * branches, dim=0).to(dtype),
                    t=time_inputs[step, :width], **guidance,
                )
                conditioned, *unguided = output.chunk(branches, dim=0)
                velocity = conditioned
                for scale, branch in zip(strengths, unguided, strict=True):
                    velocity = velocity + scale * (conditioned - branch)
            else:
                velocity = model.forward_with_encoded_conditions(
                    x_t=latent.to(dtype), t=time_inputs[step, :batch], **conditional,
                )
            if options["rescale_k"] is not None and options["rescale_sigma"] is not None:
                velocity = temporal_score_rescale(
                    v_pred=velocity, x_t=latent, t=schedule[step], rescale_k=float(options["rescale_k"]),
                    rescale_sigma=float(options["rescale_sigma"]),
                )
            latent = latent + velocity * steps[step]
        if callback is not None:
            callback(count, count)
        return latent

    return sample
