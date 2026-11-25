param(
    [Parameter(Mandatory)][string] $GptDir,
    [Parameter(Mandatory)][string] $Trtexec,
    [Parameter(Mandatory)][string] $CacheDir
)

$ErrorActionPreference = 'Stop'
$GptDir = (Resolve-Path -LiteralPath $GptDir).Path
$Trtexec = (Resolve-Path -LiteralPath $Trtexec).Path
New-Item -Path $CacheDir -ItemType Directory -Force | Out-Null
$CacheDir = (Resolve-Path -LiteralPath $CacheDir).Path

if (-not ('AliceNativeTts.GraphFingerprint' -as [type])) {
    Add-Type -TypeDefinition @'
using System.IO;

namespace AliceNativeTts {
    public static class GraphFingerprint {
        public static string ForFile(string path) {
            ulong hash = 0xcbf29ce484222325UL;
            byte[] buffer = new byte[1048576];
            using FileStream input = File.OpenRead(path);
            int size;
            while ((size = input.Read(buffer, 0, buffer.Length)) > 0) {
                for (int index = 0; index < size; ++index)
                    hash = unchecked((hash ^ buffer[index]) * 0x100000001b3UL);
            }
            return hash.ToString("x16");
        }
    }
}
'@
}

$Profiles = @(
    @{
        Name = 'gpt_encoder'
        Model = 'gpt_encoder.onnx'
        Shapes = @(
            'phoneme_ids:1x1,prompts:1x1,bert_feature:1x1024x1',
            'phoneme_ids:1x72,prompts:1x96,bert_feature:1x1024x72',
            'phoneme_ids:1x512,prompts:1x512,bert_feature:1x1024x512'
        )
    },
    @{
        Name = 'gpt_step'
        Model = 'gpt_step_active_len.onnx'
        Shapes = @('active_len:1', 'active_len:272', 'active_len:512')
    }
)

foreach ($Graph in $Profiles) {
    $Source = Join-Path $GptDir $Graph.Model
    $Fingerprint = [AliceNativeTts.GraphFingerprint]::ForFile($Source)
    $Plan = Join-Path $CacheDir "$($Graph.Name)-$Fingerprint-trt112-fp16-cache.plan"
    $Log = Join-Path $CacheDir "$($Graph.Name).log"
    if ((Test-Path -LiteralPath $Plan) -and (Get-Item -LiteralPath $Plan).Length -gt 0) {
        Write-Output $Plan
        continue
    }
    $Pending = "$Plan.$PID.tmp"
    $BuildArguments = @(
        "--onnx=$Source", "--saveEngine=$Pending", "--minShapes=$($Graph.Shapes[0])",
        "--optShapes=$($Graph.Shapes[1])", "--maxShapes=$($Graph.Shapes[2])",
        '--skipInference', '--noTF32', '--maxAuxStreams=0',
        '--builderOptimizationLevel=5', '--memPoolSize=workspace:4096'
    )
    & $Trtexec @BuildArguments *> $Log
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Pending)) {
        throw "TensorRT plan build failed. See $Log"
    }
    Move-Item -LiteralPath $Pending -Destination $Plan -Force
    Write-Output $Plan
}
