//! Expand common display forms before phonemization and sentence splitting.
use super::{chinese_integer, english_integer, is_han};
use regex::{Captures, Regex};
use std::sync::OnceLock;

fn rules() -> &'static [Regex; 5] {
    static RULES: OnceLock<[Regex; 5]> = OnceLock::new();
    RULES.get_or_init(|| {
        [
            Regex::new(
                r"([12][0-9]{3})\s*(?:-|/|年)\s*([0-9]{1,2})\s*(?:-|/|月)\s*([0-9]{1,2})(?:\s*日)?",
            )
            .unwrap(),
            Regex::new(r"([0-9]{1,2}):([0-9]{2})(?::([0-9]{2}))?").unwrap(),
            Regex::new(r"([¥￥$€£])([+-]?[0-9]+(?:\.[0-9]+)?)").unwrap(),
            Regex::new(r"([+-]?[0-9]+(?:\.[0-9]+)?)\s*[%％]").unwrap(),
            Regex::new(r"([+-]?[0-9]+(?:\.[0-9]+)?)\s*(°C|℃|°F|℉)").unwrap(),
        ]
    })
}

fn number(value: &str, zh: bool, digits_only: bool) -> String {
    let mut value = value;
    let mut out = String::new();
    if let Some(rest) = value.strip_prefix('-') {
        out.push_str(if zh { "负" } else { "minus " });
        value = rest;
    } else if let Some(rest) = value.strip_prefix('+') {
        out.push_str(if zh { "正" } else { "plus " });
        value = rest;
    }
    let (whole, fraction) = value
        .split_once('.')
        .map_or((value, None), |(a, b)| (a, Some(b)));
    let cardinal = if zh { chinese_integer } else { english_integer };
    let digit_spelling = |s: &str| {
        s.chars()
            .map(|c| cardinal(c.to_digit(10).unwrap() as u64))
            .collect::<Vec<_>>()
            .join(if zh { "" } else { " " })
    };
    out.push_str(
        &if digits_only || whole.len() > 12 || (whole.len() > 1 && whole.starts_with('0')) {
            digit_spelling(whole)
        } else {
            cardinal(whole.parse().unwrap())
        },
    );
    if let Some(fraction) = fraction {
        out.push_str(if zh { "点" } else { " point " });
        out.push_str(&digit_spelling(fraction));
    }
    out
}

fn valid_date(y: u32, m: u32, d: u32) -> bool {
    let days = match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
                29
            } else {
                28
            }
        }
        _ => 0,
    };
    d > 0 && d <= days
}

/// Idempotent: calling this on already expanded text preserves its pronunciation.
pub fn normalize(text: &str) -> String {
    let zh = text.chars().any(is_han);
    let mut text: String = text
        .chars()
        .map(|ch| match ch {
            '０'..='９' => char::from_u32(ch as u32 - '０' as u32 + '0' as u32).unwrap(),
            '：' => ':',
            '．' => '.',
            '－' | '−' => '-',
            '／' => '/',
            '＼' => '\\',
            '＝' => '=',
            other => other,
        })
        .collect();
    let rules = rules();
    text = rules[0]
        .replace_all(&text, |c: &Captures<'_>| {
            let y = c[1].parse().unwrap();
            let m = c[2].parse::<u32>().unwrap();
            let d = c[3].parse::<u32>().unwrap();
            if !valid_date(y, m, d) {
                return c[0].to_owned();
            }
            if zh {
                format!(
                    "{}年{}月{}日",
                    number(&c[1], true, true),
                    chinese_integer(m as u64),
                    chinese_integer(d as u64)
                )
            } else {
                let months = [
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                    "December",
                ];
                format!(
                    "{} {} {}",
                    months[m as usize - 1],
                    english_integer(d as u64),
                    english_integer(y as u64)
                )
            }
        })
        .into_owned();
    text = rules[1]
        .replace_all(&text, |c: &Captures<'_>| {
            let h = c[1].parse::<u64>().unwrap();
            let m = c[2].parse::<u64>().unwrap();
            let s = c.get(3).map(|s| s.as_str().parse::<u64>().unwrap());
            if h > 23 || m > 59 || s.is_some_and(|s| s > 59) {
                return c[0].to_owned();
            }
            if zh {
                format!(
                    "{}点{}{}",
                    chinese_integer(h),
                    if m == 0 {
                        "整".into()
                    } else {
                        format!("{}分", chinese_integer(m))
                    },
                    s.map_or(String::new(), |s| format!("{}秒", chinese_integer(s)))
                )
            } else {
                format!(
                    "{} {}{}",
                    english_integer(h),
                    if m == 0 {
                        "o'clock".into()
                    } else if m < 10 {
                        format!("oh {}", english_integer(m))
                    } else {
                        english_integer(m)
                    },
                    s.map_or(String::new(), |s| format!(
                        " {} seconds",
                        english_integer(s)
                    ))
                )
            }
        })
        .into_owned();
    text = rules[2]
        .replace_all(&text, |c: &Captures<'_>| {
            let unit = match (&c[1], zh) {
                ("¥" | "￥", true) => "元",
                ("$", true) => "美元",
                ("€", true) => "欧元",
                ("£", true) => "英镑",
                ("¥" | "￥", false) => " yuan",
                ("$", false) => " dollars",
                ("€", false) => " euros",
                _ => " pounds",
            };
            format!(" {}{} ", number(&c[2], zh, false), unit)
        })
        .into_owned();
    text = rules[3]
        .replace_all(&text, |c: &Captures<'_>| {
            if zh {
                format!("百分之{}", number(&c[1], zh, false))
            } else {
                format!(" {} percent ", number(&c[1], zh, false))
            }
        })
        .into_owned();
    text = rules[4]
        .replace_all(&text, |c: &Captures<'_>| {
            let celsius = matches!(&c[2], "°C" | "℃");
            if zh {
                format!(
                    "{}{}度",
                    if celsius { "摄氏" } else { "华氏" },
                    number(&c[1], zh, false)
                )
            } else {
                format!(
                    " {} degrees {} ",
                    number(&c[1], zh, false),
                    if celsius { "Celsius" } else { "Fahrenheit" }
                )
            }
        })
        .into_owned();
    let mut chars = text.chars().peekable();
    let mut out = String::new();
    while let Some(ch) = chars.next() {
        if ch.is_ascii_digit()
            || (matches!(ch, '-' | '+') && chars.peek().is_some_and(|c| c.is_ascii_digit()))
        {
            let mut value = ch.to_string();
            while chars.peek().is_some_and(|c| c.is_ascii_digit()) {
                value.push(chars.next().unwrap());
            }
            if chars.peek() == Some(&'.') {
                let mut rest = chars.clone();
                rest.next();
                if rest.peek().is_some_and(|c| c.is_ascii_digit()) {
                    value.push(chars.next().unwrap());
                    while chars.peek().is_some_and(|c| c.is_ascii_digit()) {
                        value.push(chars.next().unwrap());
                    }
                }
            }
            out.push(' ');
            out.push_str(&number(&value, zh, false));
            out.push(' ');
            continue;
        }
        // Equality is meaningful speech content, but is not a phoneme token.
        if ch == '=' {
            out.push_str(if zh { "等于" } else { " equals " });
            continue;
        }
        // Date separators were expanded above. Remaining separators occur
        // in alternatives and file paths, outside the phoneme alphabet.
        if matches!(ch, '/' | '\\') {
            out.push_str(match (ch, zh) {
                ('/', true) => "斜杠",
                ('/', false) => " slash ",
                (_, true) => "反斜杠",
                (_, false) => " backslash ",
            });
            continue;
        }
        out.push(match ch {
            '，' | '、' => ',',
            '。' | ';' | '；' | ':' | '：' => '.',
            '！' => '!',
            '？' => '?',
            '–' | '—' => '-',
            '“' | '”' | '"' | '(' | ')' | '（' | '）' => ' ',
            '’' | '‘' => '\'',
            other => other,
        });
    }
    if !out.trim_end().ends_with(['.', '!', '?']) {
        out.push('.');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn compact(s: &str) -> String {
        s.chars().filter(|c| !c.is_whitespace()).collect()
    }
    #[test]
    fn dates_times_and_fullwidth_numbers() {
        assert_eq!(
            compact(&normalize(
                "今天是２０２６-１０-０３，１４：０５：０９出发。"
            )),
            "今天是二零二六年十月三日,十四点五分九秒出发."
        );
        assert!(valid_date(2024, 2, 29));
        assert!(!valid_date(2026, 2, 29));
        assert_eq!(
            normalize("Meet on 2026/10/03 at 09:05."),
            "Meet on October three two thousand twenty six at nine oh five."
        );
    }
    #[test]
    fn amounts_percentages_and_signed_temperature() {
        assert_eq!(
            compact(&normalize("费用￥12.5，优惠20%，气温-3.5℃，编号007。")),
            "费用十二点五元,优惠百分之二十,气温摄氏负三点五度,编号零零七."
        );
        assert!(
            normalize("Pay $12.5 with 20% off at -3.5°C.").contains("twelve point five dollars")
        );
        assert!(normalize("Pay $12.5 with 20% off at -3.5°C.")
            .contains("minus three point five degrees Celsius"));
    }
    #[test]
    fn expansion_is_idempotent_and_keeps_decimal() {
        for text in [
            "今天2026-10-03，费用￥12.5。",
            "Test 42.",
            "Temperature -3.5°C.",
            "选择CPU／GPU，文件src/audio。",
            "Use src/audio and CPU/GPU.",
            r"Open C:\audio\sample.wav.",
            "结果x=2，y＝-3.5。",
            "Value x=2 and y=-3.5.",
        ] {
            let expanded = normalize(text);
            assert_eq!(normalize(&expanded), expanded);
        }
    }

    #[test]
    fn slashes_in_alternatives_and_paths_are_spoken() {
        assert_eq!(
            compact(&normalize("选择CPU／GPU，文件src/audio。")),
            "选择CPU斜杠GPU,文件src斜杠audio."
        );
        assert_eq!(
            normalize("Use src/audio and CPU/GPU."),
            "Use src slash audio and CPU slash GPU."
        );
        assert_eq!(
            normalize(r"Open C:\audio\sample.wav."),
            "Open C. backslash audio backslash sample.wav."
        );
        assert_eq!(
            compact(&normalize(r"目录C:\audio＼sample.wav。")),
            "目录C.反斜杠audio反斜杠sample.wav."
        );
    }

    #[test]
    fn dates_are_expanded_before_remaining_slashes() {
        let text = normalize("日期2026/10/03，选择CPU/GPU。错误日期2026/02/30。");
        assert!(text.contains("二零二六年十月三日"));
        assert!(text.contains("CPU斜杠GPU"));
        assert!(!text.contains("二零二六年二月三十日"));
        assert!(!text.contains('/'));
    }

    #[test]
    fn equality_is_spoken_without_losing_operands_or_signs() {
        assert_eq!(
            compact(&normalize("结果x=2，y＝-3.5。")),
            "结果x等于二,y等于负三点五."
        );
        assert_eq!(compact(&normalize("公式E=mc。")), "公式E等于mc.");
        assert_eq!(normalize("x=y."), "x equals y.");
        assert_eq!(compact(&normalize("结果＝２。")), "结果等于二.");
    }
}
