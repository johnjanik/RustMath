//! Buzzard's algorithm for computing modular forms

/// Buzzard's overconvergent modular symbols algorithm
#[derive(Clone, Debug)]
pub struct BuzzardAlgorithm {
    level: u64,
    weight: i64,
}

impl BuzzardAlgorithm {
    pub fn new(level: u64, weight: i64) -> Self {
        Self { level, weight }
    }

    pub fn level(&self) -> u64 {
        self.level
    }

    pub fn weight(&self) -> i64 {
        self.weight
    }

    /// The ordinary projection e = lim U_p^{n!} on the space of overconvergent
    /// modular forms of this level and weight.
    ///
    /// NOT IMPLEMENTED.  Buzzard's method needs the matrix of the U_p operator on
    /// a basis of overconvergent forms of the given level and weight, over Z_p to
    /// a controlled precision, and then Hida's idempotent as the limit of
    /// U_p^{n!}.  None of that exists in this crate: there is no overconvergent
    /// modular forms space (`crate::overconvergent` is a stub), no U_p matrix on
    /// one, and no p to iterate at -- `BuzzardAlgorithm` carries only a level and
    /// a weight.
    ///
    /// This used to return an empty `Vec`, which reads as "the ordinary part is
    /// zero" and is wrong for every ordinary form.
    pub fn ordinary_projection(&self) -> Vec<f64> {
        unimplemented!(
            "BuzzardAlgorithm::ordinary_projection (level {}, weight {}): not implemented. \
             It needs the U_p matrix on a basis of overconvergent modular forms over Z_p \
             and Hida's idempotent lim U_p^(n!); this crate has no overconvergent forms \
             space, no U_p on one, and BuzzardAlgorithm does not even carry the prime p. \
             Previously returned an empty vector, i.e. a silent \"the ordinary part is zero\".",
            self.level, self.weight
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buzzard_data() {
        let b = BuzzardAlgorithm::new(11, 2);
        assert_eq!(b.level(), 11);
        assert_eq!(b.weight(), 2);
    }

    /// The ordinary projection is refused, not faked.  An empty vector reads as
    /// "the ordinary part is zero", which is wrong for every ordinary form.
    #[test]
    #[should_panic(expected = "BuzzardAlgorithm::ordinary_projection")]
    fn test_ordinary_projection_is_refused_not_faked() {
        let _ = BuzzardAlgorithm::new(11, 2).ordinary_projection();
    }
}
