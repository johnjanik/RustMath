//! String operations for string monoids

/// Compute the longest common prefix of two strings
pub fn longest_common_prefix(s1: &str, s2: &str) -> String {
    s1.chars()
        .zip(s2.chars())
        .take_while(|(c1, c2)| c1 == c2)
        .map(|(c, _)| c)
        .collect()
}

/// Compute the longest common suffix of two strings
pub fn longest_common_suffix(s1: &str, s2: &str) -> String {
    s1.chars()
        .rev()
        .zip(s2.chars().rev())
        .take_while(|(c1, c2)| c1 == c2)
        .map(|(c, _)| c)
        .collect::<String>()
        .chars()
        .rev()
        .collect()
}

/// Check if s1 is a prefix of s2
pub fn is_prefix(s1: &str, s2: &str) -> bool {
    s2.starts_with(s1)
}

/// Check if s1 is a suffix of s2
pub fn is_suffix(s1: &str, s2: &str) -> bool {
    s2.ends_with(s1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_longest_common_prefix() {
        assert_eq!(longest_common_prefix("hello", "help"), "hel");
        assert_eq!(longest_common_prefix("abc", "def"), "");
    }

    #[test]
    fn test_longest_common_suffix() {
        assert_eq!(longest_common_suffix("testing", "running"), "ing");
        assert_eq!(longest_common_suffix("abc", "def"), "");
    }
}
