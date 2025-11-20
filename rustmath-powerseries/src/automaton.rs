//! Weighted automata for recognizable series
//!
//! A weighted automaton is a finite automaton where transitions carry weights from a semiring.
//! The recognizable series (or rational series) is the formal power series whose coefficient
//! at x^n is the sum of weights of all accepting paths of length n.
//!
//! # Theory
//!
//! A weighted automaton A over a semiring R consists of:
//! - A finite set of states Q = {0, 1, ..., n-1}
//! - A finite alphabet Σ = {0, 1, ..., m-1}
//! - Initial weights: λ: Q → R (vector of length n)
//! - Final weights: γ: Q → R (vector of length n)
//! - Transition function: δ: Q × Σ × Q → R
//!
//! The weight of a path p = (q₀, a₁, q₁, a₂, ..., aₙ, qₙ) is:
//! w(p) = λ(q₀) × δ(q₀, a₁, q₁) × ... × δ(qₙ₋₁, aₙ, qₙ) × γ(qₙ)
//!
//! The coefficient of x^n in the recognizable series is the sum of weights
//! of all paths of length n.

use rustmath_core::Ring;
use std::collections::HashMap;

use crate::PowerSeries;

/// A weighted finite automaton over a semiring R
///
/// States are represented as indices 0, 1, ..., num_states-1
/// Alphabet symbols are represented as indices 0, 1, ..., alphabet_size-1
#[derive(Clone, Debug)]
pub struct WeightedAutomaton<R: Ring> {
    /// Number of states
    num_states: usize,
    /// Size of the alphabet
    alphabet_size: usize,
    /// Initial weights for each state (length = num_states)
    initial_weights: Vec<R>,
    /// Final weights for each state (length = num_states)
    final_weights: Vec<R>,
    /// Transitions: (from_state, symbol, to_state) -> weight
    /// Only non-zero transitions are stored
    transitions: HashMap<(usize, usize, usize), R>,
}

impl<R: Ring> WeightedAutomaton<R> {
    /// Create a new weighted automaton with the given number of states and alphabet size
    ///
    /// All initial and final weights are set to zero by default.
    pub fn new(num_states: usize, alphabet_size: usize) -> Self {
        WeightedAutomaton {
            num_states,
            alphabet_size,
            initial_weights: vec![R::zero(); num_states],
            final_weights: vec![R::zero(); num_states],
            transitions: HashMap::new(),
        }
    }

    /// Set the initial weight for a state
    pub fn set_initial_weight(&mut self, state: usize, weight: R) {
        if state < self.num_states {
            self.initial_weights[state] = weight;
        }
    }

    /// Set the final weight for a state
    pub fn set_final_weight(&mut self, state: usize, weight: R) {
        if state < self.num_states {
            self.final_weights[state] = weight;
        }
    }

    /// Add a transition with a weight
    ///
    /// If the transition already exists, its weight is replaced.
    pub fn add_transition(&mut self, from: usize, symbol: usize, to: usize, weight: R) {
        if from < self.num_states && to < self.num_states && symbol < self.alphabet_size {
            if !weight.is_zero() {
                self.transitions.insert((from, symbol, to), weight);
            } else {
                self.transitions.remove(&(from, symbol, to));
            }
        }
    }

    /// Get the weight of a transition (returns zero if not found)
    pub fn transition_weight(&self, from: usize, symbol: usize, to: usize) -> R {
        self.transitions
            .get(&(from, symbol, to))
            .cloned()
            .unwrap_or_else(|| R::zero())
    }

    /// Get the number of states
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// Get the alphabet size
    pub fn alphabet_size(&self) -> usize {
        self.alphabet_size
    }

    /// Compute the recognizable series up to the given precision
    ///
    /// The coefficient of x^n is the sum of weights of all accepting paths of length n.
    ///
    /// # Algorithm
    ///
    /// Uses dynamic programming:
    /// - weight[0][q] = λ(q) (initial weight)
    /// - weight[n+1][q'] = Σ_q Σ_a weight[n][q] × δ(q, a, q')
    /// - coeff[n] = Σ_q weight[n][q] × γ(q) (final weight)
    pub fn recognizable_series(&self, precision: usize) -> PowerSeries<R> {
        let mut coefficients = Vec::with_capacity(precision);

        // Current weights at each state (initially, the initial weights)
        let mut current_weights = self.initial_weights.clone();

        for length in 0..precision {
            // Compute coefficient for this length: sum over all states
            let mut coeff = R::zero();
            for state in 0..self.num_states {
                coeff = coeff + current_weights[state].clone() * self.final_weights[state].clone();
            }
            coefficients.push(coeff);

            // Prepare next weights if we haven't reached precision yet
            if length + 1 < precision {
                let mut next_weights = vec![R::zero(); self.num_states];

                // For each state and symbol, propagate weights
                for ((from, _symbol, to), trans_weight) in &self.transitions {
                    let contribution = current_weights[*from].clone() * trans_weight.clone();
                    next_weights[*to] = next_weights[*to].clone() + contribution;
                }

                current_weights = next_weights;
            }
        }

        PowerSeries::new(coefficients, precision)
    }

    /// Create a simple automaton that recognizes words over a single-letter alphabet
    /// with geometric weights
    ///
    /// This creates the series: initial * (1 + weight*x + weight^2*x^2 + ...)
    /// = initial / (1 - weight*x) as a truncated power series
    pub fn geometric(initial: R, weight: R, precision: usize) -> PowerSeries<R> {
        let mut automaton = WeightedAutomaton::new(1, 1);
        automaton.set_initial_weight(0, initial.clone());
        automaton.set_final_weight(0, R::one());
        automaton.add_transition(0, 0, 0, weight);

        automaton.recognizable_series(precision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_empty_automaton() {
        let automaton: WeightedAutomaton<Integer> = WeightedAutomaton::new(1, 1);
        let series = automaton.recognizable_series(5);

        // No initial or final weights, so all coefficients should be zero
        for i in 0..5 {
            assert_eq!(series.coeff(i), &Integer::from(0));
        }
    }

    #[test]
    fn test_single_state_constant() {
        // Automaton that accepts only the empty word with weight 5
        let mut automaton = WeightedAutomaton::new(1, 1);
        automaton.set_initial_weight(0, Integer::from(1));
        automaton.set_final_weight(0, Integer::from(5));

        let series = automaton.recognizable_series(5);

        // Only x^0 term should be non-zero
        assert_eq!(series.coeff(0), &Integer::from(5));
        assert_eq!(series.coeff(1), &Integer::from(0));
        assert_eq!(series.coeff(2), &Integer::from(0));
    }

    #[test]
    fn test_geometric_series() {
        // Automaton for 1 + x + x^2 + x^3 + ...
        // One state, self-loop with weight 1
        let mut automaton = WeightedAutomaton::new(1, 1);
        automaton.set_initial_weight(0, Integer::from(1));
        automaton.set_final_weight(0, Integer::from(1));
        automaton.add_transition(0, 0, 0, Integer::from(1));

        let series = automaton.recognizable_series(6);

        // All coefficients should be 1
        for i in 0..6 {
            assert_eq!(series.coeff(i), &Integer::from(1));
        }
    }

    #[test]
    fn test_geometric_helper() {
        // Test the geometric helper function
        let series = WeightedAutomaton::geometric(Integer::from(1), Integer::from(1), 6);

        for i in 0..6 {
            assert_eq!(series.coeff(i), &Integer::from(1));
        }
    }

    #[test]
    fn test_weighted_geometric() {
        // Automaton for 2 + 2x + 2x^2 + ...
        let mut automaton = WeightedAutomaton::new(1, 1);
        automaton.set_initial_weight(0, Integer::from(1));
        automaton.set_final_weight(0, Integer::from(2));
        automaton.add_transition(0, 0, 0, Integer::from(1));

        let series = automaton.recognizable_series(6);

        for i in 0..6 {
            assert_eq!(series.coeff(i), &Integer::from(2));
        }
    }

    #[test]
    fn test_fibonacci_series() {
        // Automaton for Fibonacci series: 1, 1, 2, 3, 5, 8, ...
        // Two states: state 0 and state 1
        // Initial: both states have weight 1
        // Final: state 0 has weight 1, state 1 has weight 0
        // Transitions: 0->1 (symbol 0, weight 1), 1->0 (symbol 0, weight 1)
        //              1->1 (symbol 0, weight 1)
        let mut automaton = WeightedAutomaton::new(2, 1);
        automaton.set_initial_weight(0, Integer::from(1));
        automaton.set_initial_weight(1, Integer::from(1));
        automaton.set_final_weight(0, Integer::from(1));
        automaton.set_final_weight(1, Integer::from(0));

        automaton.add_transition(0, 0, 1, Integer::from(1));
        automaton.add_transition(1, 0, 0, Integer::from(1));
        automaton.add_transition(1, 0, 1, Integer::from(1));

        let series = automaton.recognizable_series(10);

        // Fibonacci sequence
        assert_eq!(series.coeff(0), &Integer::from(1)); // F(1)
        assert_eq!(series.coeff(1), &Integer::from(1)); // F(2)
        assert_eq!(series.coeff(2), &Integer::from(2)); // F(3)
        assert_eq!(series.coeff(3), &Integer::from(3)); // F(4)
        assert_eq!(series.coeff(4), &Integer::from(5)); // F(5)
        assert_eq!(series.coeff(5), &Integer::from(8)); // F(6)
        assert_eq!(series.coeff(6), &Integer::from(13)); // F(7)
        assert_eq!(series.coeff(7), &Integer::from(21)); // F(8)
    }

    #[test]
    fn test_two_state_alternating() {
        // Automaton that alternates between two states
        // This gives series with alternating coefficients
        let mut automaton = WeightedAutomaton::new(2, 1);
        automaton.set_initial_weight(0, Integer::from(1));
        automaton.set_final_weight(0, Integer::from(1));
        automaton.set_final_weight(1, Integer::from(2));

        // 0 -> 1 and 1 -> 0
        automaton.add_transition(0, 0, 1, Integer::from(1));
        automaton.add_transition(1, 0, 0, Integer::from(1));

        let series = automaton.recognizable_series(6);

        assert_eq!(series.coeff(0), &Integer::from(1)); // Start at 0
        assert_eq!(series.coeff(1), &Integer::from(2)); // Move to 1
        assert_eq!(series.coeff(2), &Integer::from(1)); // Back to 0
        assert_eq!(series.coeff(3), &Integer::from(2)); // Move to 1
        assert_eq!(series.coeff(4), &Integer::from(1)); // Back to 0
    }

    #[test]
    fn test_multiple_paths() {
        // Automaton with multiple paths of the same length
        // Two states, both transitions available
        let mut automaton = WeightedAutomaton::new(2, 1);
        automaton.set_initial_weight(0, Integer::from(1));
        automaton.set_final_weight(0, Integer::from(1));
        automaton.set_final_weight(1, Integer::from(1));

        // Self-loops on both states
        automaton.add_transition(0, 0, 0, Integer::from(1));
        automaton.add_transition(0, 0, 1, Integer::from(1));
        automaton.add_transition(1, 0, 0, Integer::from(1));
        automaton.add_transition(1, 0, 1, Integer::from(1));

        let series = automaton.recognizable_series(5);

        // Length 0: just state 0, weight 1
        assert_eq!(series.coeff(0), &Integer::from(1));
        // Length 1: from 0 can go to 0 or 1, both count, so 2
        assert_eq!(series.coeff(1), &Integer::from(2));
        // Length 2: 4 paths (each of the 2 states at step 1 can go to 2 states)
        assert_eq!(series.coeff(2), &Integer::from(4));
        // Length 3: 8 paths
        assert_eq!(series.coeff(3), &Integer::from(8));
        // Pattern: powers of 2
        assert_eq!(series.coeff(4), &Integer::from(16));
    }
}
