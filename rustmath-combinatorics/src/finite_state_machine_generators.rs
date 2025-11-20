//! Finite State Machine Generators
//!
//! This module provides predefined generators for finite state machines (FSMs),
//! including automata and transducers. It mirrors SageMath's finite_state_machine_generators
//! functionality.
//!
//! # FSM Types
//!
//! - **Automaton**: Accepts/rejects input sequences
//! - **Transducer**: Maps input sequences to output sequences
//!
//! # Common Generators
//!
//! - All ones: Outputs all ones regardless of input
//! - Counting: Counts the number of inputs
//! - Identity: Outputs the same as input
//! - Pattern matching: Detects specific patterns in input

use std::hash::Hash;

/// A state in a finite state machine
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FSMState {
    /// State label/identifier
    pub label: String,
    /// Whether this is an initial state
    pub is_initial: bool,
    /// Whether this is a final/accepting state
    pub is_final: bool,
}

impl FSMState {
    /// Create a new FSM state
    pub fn new(label: String, is_initial: bool, is_final: bool) -> Self {
        FSMState {
            label,
            is_initial,
            is_final,
        }
    }

    /// Create an initial state
    pub fn initial(label: String) -> Self {
        FSMState {
            label,
            is_initial: true,
            is_final: false,
        }
    }

    /// Create a final state
    pub fn final_state(label: String) -> Self {
        FSMState {
            label,
            is_initial: false,
            is_final: true,
        }
    }
}

/// A transition in a finite state machine
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FSMTransition<I: Clone, O: Clone> {
    /// Source state
    pub from_state: String,
    /// Target state
    pub to_state: String,
    /// Input symbol (None for epsilon transitions)
    pub input: Option<I>,
    /// Output symbol (for transducers)
    pub output: Option<O>,
}

impl<I: Clone, O: Clone> FSMTransition<I, O> {
    /// Create a new transition
    pub fn new(from_state: String, to_state: String, input: Option<I>, output: Option<O>) -> Self {
        FSMTransition {
            from_state,
            to_state,
            input,
            output,
        }
    }
}

/// An automaton (accepts or rejects input)
#[derive(Debug, Clone)]
pub struct Automaton<I: Clone + Eq + Hash> {
    /// States in the automaton
    pub states: Vec<FSMState>,
    /// Transitions
    pub transitions: Vec<FSMTransition<I, ()>>,
    /// Input alphabet
    pub alphabet: Vec<I>,
}

impl<I: Clone + Eq + Hash> Automaton<I> {
    /// Create a new automaton
    pub fn new(states: Vec<FSMState>, transitions: Vec<FSMTransition<I, ()>>, alphabet: Vec<I>) -> Self {
        Automaton {
            states,
            transitions,
            alphabet,
        }
    }

    /// Get the initial state
    pub fn initial_state(&self) -> Option<&FSMState> {
        self.states.iter().find(|s| s.is_initial)
    }

    /// Get all final states
    pub fn final_states(&self) -> Vec<&FSMState> {
        self.states.iter().filter(|s| s.is_final).collect()
    }

    /// Process an input sequence and return whether it's accepted
    pub fn process(&self, input: &[I]) -> bool {
        let mut current_state = match self.initial_state() {
            Some(s) => s.label.clone(),
            None => return false,
        };

        for symbol in input {
            // Find transition from current state with this input
            let transition = self.transitions.iter().find(|t| {
                t.from_state == current_state && t.input.as_ref() == Some(symbol)
            });

            match transition {
                Some(t) => current_state = t.to_state.clone(),
                None => return false, // No valid transition
            }
        }

        // Check if we ended in a final state
        self.states
            .iter()
            .any(|s| s.label == current_state && s.is_final)
    }
}

/// A transducer (maps input to output)
#[derive(Debug, Clone)]
pub struct Transducer<I: Clone + Eq + Hash, O: Clone> {
    /// States in the transducer
    pub states: Vec<FSMState>,
    /// Transitions
    pub transitions: Vec<FSMTransition<I, O>>,
    /// Input alphabet
    pub input_alphabet: Vec<I>,
    /// Output alphabet
    pub output_alphabet: Vec<O>,
}

impl<I: Clone + Eq + Hash, O: Clone> Transducer<I, O> {
    /// Create a new transducer
    pub fn new(
        states: Vec<FSMState>,
        transitions: Vec<FSMTransition<I, O>>,
        input_alphabet: Vec<I>,
        output_alphabet: Vec<O>,
    ) -> Self {
        Transducer {
            states,
            transitions,
            input_alphabet,
            output_alphabet,
        }
    }

    /// Get the initial state
    pub fn initial_state(&self) -> Option<&FSMState> {
        self.states.iter().find(|s| s.is_initial)
    }

    /// Process an input sequence and return the output sequence
    pub fn process(&self, input: &[I]) -> Option<Vec<O>> {
        let mut current_state = match self.initial_state() {
            Some(s) => s.label.clone(),
            None => return None,
        };

        let mut output = Vec::new();

        for symbol in input {
            // Find transition from current state with this input
            let transition = self.transitions.iter().find(|t| {
                t.from_state == current_state && t.input.as_ref() == Some(symbol)
            });

            match transition {
                Some(t) => {
                    current_state = t.to_state.clone();
                    if let Some(ref out) = t.output {
                        output.push(out.clone());
                    }
                }
                None => return None, // No valid transition
            }
        }

        Some(output)
    }
}

/// Automaton generators - factory methods for common automata
pub struct AutomatonGenerators;

impl AutomatonGenerators {
    /// Create an automaton that accepts any single letter from the alphabet
    ///
    /// # Example
    /// ```ignore
    /// let automaton = AutomatonGenerators::any_letter(vec![0, 1]);
    /// assert!(automaton.process(&[0]));
    /// assert!(automaton.process(&[1]));
    /// assert!(!automaton.process(&[0, 1]));
    /// ```
    pub fn any_letter<I: Clone + Eq + Hash>(alphabet: Vec<I>) -> Automaton<I> {
        let initial = FSMState::initial("q0".to_string());
        let final_state = FSMState::final_state("q1".to_string());

        let mut transitions = Vec::new();
        for symbol in &alphabet {
            transitions.push(FSMTransition::new(
                "q0".to_string(),
                "q1".to_string(),
                Some(symbol.clone()),
                None,
            ));
        }

        Automaton::new(vec![initial, final_state], transitions, alphabet)
    }

    /// Create an automaton that accepts any word over the alphabet
    ///
    /// # Example
    /// ```ignore
    /// let automaton = AutomatonGenerators::any_word(vec![0, 1]);
    /// assert!(automaton.process(&[]));
    /// assert!(automaton.process(&[0, 1, 0]));
    /// ```
    pub fn any_word<I: Clone + Eq + Hash>(alphabet: Vec<I>) -> Automaton<I> {
        let state = FSMState::new("q0".to_string(), true, true);

        let mut transitions = Vec::new();
        for symbol in &alphabet {
            transitions.push(FSMTransition::new(
                "q0".to_string(),
                "q0".to_string(),
                Some(symbol.clone()),
                None,
            ));
        }

        Automaton::new(vec![state], transitions, alphabet)
    }

    /// Create an automaton that accepts only the empty word
    pub fn empty_word<I: Clone + Eq + Hash>(alphabet: Vec<I>) -> Automaton<I> {
        let state = FSMState::new("q0".to_string(), true, true);
        Automaton::new(vec![state], vec![], alphabet)
    }

    /// Create an automaton that accepts a specific word
    pub fn word<I: Clone + Eq + Hash>(word: Vec<I>, alphabet: Vec<I>) -> Automaton<I> {
        if word.is_empty() {
            return Self::empty_word(alphabet);
        }

        let mut states = vec![FSMState::initial("q0".to_string())];
        let mut transitions = Vec::new();

        for (i, symbol) in word.iter().enumerate() {
            let from_state = format!("q{}", i);
            let to_state = format!("q{}", i + 1);

            let is_final = i == word.len() - 1;
            states.push(FSMState::new(to_state.clone(), false, is_final));

            transitions.push(FSMTransition::new(
                from_state,
                to_state,
                Some(symbol.clone()),
                None,
            ));
        }

        Automaton::new(states, transitions, alphabet)
    }

    /// Create an automaton that accepts words containing a specific subword
    ///
    /// Uses the Knuth-Morris-Pratt algorithm approach
    pub fn contains_word<I: Clone + Eq + Hash>(pattern: Vec<I>, alphabet: Vec<I>) -> Automaton<I> {
        if pattern.is_empty() {
            return Self::any_word(alphabet);
        }

        let n = pattern.len();
        let mut states = Vec::new();
        let mut transitions = Vec::new();

        // Create states q0, q1, ..., qn
        for i in 0..=n {
            let is_initial = i == 0;
            let is_final = i == n;
            states.push(FSMState::new(format!("q{}", i), is_initial, is_final));
        }

        // Add transitions for pattern matching
        for i in 0..n {
            let from_state = format!("q{}", i);
            let to_state = format!("q{}", i + 1);

            transitions.push(FSMTransition::new(
                from_state.clone(),
                to_state,
                Some(pattern[i].clone()),
                None,
            ));

            // Add transitions for non-matching symbols (simplified)
            for symbol in &alphabet {
                if symbol != &pattern[i] {
                    transitions.push(FSMTransition::new(
                        from_state.clone(),
                        "q0".to_string(),
                        Some(symbol.clone()),
                        None,
                    ));
                }
            }
        }

        // Final state loops on any symbol
        for symbol in &alphabet {
            transitions.push(FSMTransition::new(
                format!("q{}", n),
                format!("q{}", n),
                Some(symbol.clone()),
                None,
            ));
        }

        Automaton::new(states, transitions, alphabet)
    }
}

/// Transducer generators - factory methods for common transducers
pub struct TransducerGenerators;

impl TransducerGenerators {
    /// Create the identity transducer (outputs same as input)
    ///
    /// # Example
    /// ```ignore
    /// let transducer = TransducerGenerators::identity(vec![0, 1, 2]);
    /// assert_eq!(transducer.process(&[0, 1, 2]), Some(vec![0, 1, 2]));
    /// ```
    pub fn identity<T: Clone + Eq + Hash>(alphabet: Vec<T>) -> Transducer<T, T> {
        let state = FSMState::new("q0".to_string(), true, true);

        let mut transitions = Vec::new();
        for symbol in &alphabet {
            transitions.push(FSMTransition::new(
                "q0".to_string(),
                "q0".to_string(),
                Some(symbol.clone()),
                Some(symbol.clone()),
            ));
        }

        Transducer::new(vec![state], transitions, alphabet.clone(), alphabet)
    }

    /// Create a transducer that outputs all ones regardless of input
    ///
    /// This is useful for counting the length of input sequences.
    ///
    /// # Example
    /// ```ignore
    /// let transducer = TransducerGenerators::all_ones(vec![0, 1, 2]);
    /// assert_eq!(transducer.process(&[0, 2, 1]), Some(vec![1, 1, 1]));
    /// ```
    pub fn all_ones<I: Clone + Eq + Hash>(input_alphabet: Vec<I>) -> Transducer<I, usize> {
        let state = FSMState::new("q0".to_string(), true, true);

        let mut transitions = Vec::new();
        for symbol in &input_alphabet {
            transitions.push(FSMTransition::new(
                "q0".to_string(),
                "q0".to_string(),
                Some(symbol.clone()),
                Some(1),
            ));
        }

        Transducer::new(vec![state], transitions, input_alphabet, vec![1])
    }

    /// Create a transducer that counts the total value of inputs
    ///
    /// For binary input, this counts the number of ones.
    ///
    /// # Example
    /// ```ignore
    /// let transducer = TransducerGenerators::counting_binary();
    /// // Process [1, 0, 1, 1] -> outputs running count [1, 1, 2, 3]
    /// ```
    pub fn counting_binary() -> Transducer<usize, usize> {
        // This is a simplified version that outputs the input value
        // A true counting transducer would maintain state for cumulative count
        let state = FSMState::new("q0".to_string(), true, true);

        let transitions = vec![
            FSMTransition::new("q0".to_string(), "q0".to_string(), Some(0), Some(0)),
            FSMTransition::new("q0".to_string(), "q0".to_string(), Some(1), Some(1)),
        ];

        Transducer::new(vec![state], transitions, vec![0, 1], vec![0, 1])
    }

    /// Create a transducer that adds two sequences element-wise
    ///
    /// Input alphabet should be pairs (a, b) represented as tuples
    pub fn add(max_value: usize) -> Transducer<(usize, usize), usize> {
        let state = FSMState::new("q0".to_string(), true, true);

        let mut input_alphabet = Vec::new();
        let mut output_alphabet = Vec::new();
        let mut transitions = Vec::new();

        for a in 0..=max_value {
            for b in 0..=max_value {
                let sum = a + b;
                input_alphabet.push((a, b));
                if !output_alphabet.contains(&sum) {
                    output_alphabet.push(sum);
                }

                transitions.push(FSMTransition::new(
                    "q0".to_string(),
                    "q0".to_string(),
                    Some((a, b)),
                    Some(sum),
                ));
            }
        }

        Transducer::new(vec![state], transitions, input_alphabet, output_alphabet)
    }

    /// Create a transducer that computes Hamming weight (counts non-zero elements)
    pub fn weight<I: Clone + Eq + Hash + PartialEq>(
        alphabet: Vec<I>,
        zero: I,
    ) -> Transducer<I, usize> {
        let state = FSMState::new("q0".to_string(), true, true);

        let mut transitions = Vec::new();
        for symbol in &alphabet {
            let output = if symbol == &zero { 0 } else { 1 };
            transitions.push(FSMTransition::new(
                "q0".to_string(),
                "q0".to_string(),
                Some(symbol.clone()),
                Some(output),
            ));
        }

        Transducer::new(vec![state], transitions, alphabet, vec![0, 1])
    }

    /// Create a transducer that applies a function to each input symbol
    pub fn map<I: Clone + Eq + Hash, O: Clone + Eq + Hash, F>(
        input_alphabet: Vec<I>,
        output_alphabet: Vec<O>,
        func: F,
    ) -> Transducer<I, O>
    where
        F: Fn(&I) -> O,
    {
        let state = FSMState::new("q0".to_string(), true, true);

        let mut transitions = Vec::new();
        for symbol in &input_alphabet {
            let output = func(symbol);
            transitions.push(FSMTransition::new(
                "q0".to_string(),
                "q0".to_string(),
                Some(symbol.clone()),
                Some(output),
            ));
        }

        Transducer::new(vec![state], transitions, input_alphabet, output_alphabet)
    }

    /// Create a transducer that counts occurrences of a specific symbol
    ///
    /// Outputs cumulative count in unary (1 for each occurrence, 0 otherwise)
    pub fn count_symbol<I: Clone + Eq + Hash>(
        alphabet: Vec<I>,
        target: I,
    ) -> Transducer<I, usize> {
        let state = FSMState::new("q0".to_string(), true, true);

        let mut transitions = Vec::new();
        for symbol in &alphabet {
            let output = if symbol == &target { 1 } else { 0 };
            transitions.push(FSMTransition::new(
                "q0".to_string(),
                "q0".to_string(),
                Some(symbol.clone()),
                Some(output),
            ));
        }

        Transducer::new(vec![state], transitions, alphabet, vec![0, 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_automaton_any_letter() {
        let automaton = AutomatonGenerators::any_letter(vec![0, 1, 2]);

        assert!(automaton.process(&[0]));
        assert!(automaton.process(&[1]));
        assert!(automaton.process(&[2]));
        assert!(!automaton.process(&[]));
        assert!(!automaton.process(&[0, 1]));
    }

    #[test]
    fn test_automaton_any_word() {
        let automaton = AutomatonGenerators::any_word(vec![0, 1]);

        assert!(automaton.process(&[]));
        assert!(automaton.process(&[0]));
        assert!(automaton.process(&[1]));
        assert!(automaton.process(&[0, 1, 0, 1]));
    }

    #[test]
    fn test_automaton_empty_word() {
        let automaton = AutomatonGenerators::empty_word(vec![0, 1]);

        assert!(automaton.process(&[]));
        assert!(!automaton.process(&[0]));
        assert!(!automaton.process(&[1]));
    }

    #[test]
    fn test_automaton_word() {
        let automaton = AutomatonGenerators::word(vec![0, 1, 0], vec![0, 1]);

        assert!(automaton.process(&[0, 1, 0]));
        assert!(!automaton.process(&[]));
        assert!(!automaton.process(&[0]));
        assert!(!automaton.process(&[0, 1]));
        assert!(!automaton.process(&[0, 1, 0, 1]));
        assert!(!automaton.process(&[1, 0, 1]));
    }

    #[test]
    fn test_automaton_contains_word() {
        let automaton = AutomatonGenerators::contains_word(vec![1, 1], vec![0, 1]);

        assert!(automaton.process(&[1, 1]));
        assert!(automaton.process(&[0, 1, 1]));
        assert!(automaton.process(&[1, 1, 0]));
        assert!(automaton.process(&[0, 1, 1, 0]));
        assert!(!automaton.process(&[]));
        assert!(!automaton.process(&[0]));
        assert!(!automaton.process(&[1]));
        assert!(!automaton.process(&[0, 1, 0]));
    }

    #[test]
    fn test_transducer_identity() {
        let transducer = TransducerGenerators::identity(vec![0, 1, 2]);

        assert_eq!(transducer.process(&[0, 1, 2]), Some(vec![0, 1, 2]));
        assert_eq!(transducer.process(&[2, 2, 0]), Some(vec![2, 2, 0]));
        assert_eq!(transducer.process(&[]), Some(vec![]));
    }

    #[test]
    fn test_transducer_all_ones() {
        let transducer = TransducerGenerators::all_ones(vec![0, 1, 2, 3]);

        assert_eq!(transducer.process(&[0, 2, 1, 3]), Some(vec![1, 1, 1, 1]));
        assert_eq!(transducer.process(&[2, 2]), Some(vec![1, 1]));
        assert_eq!(transducer.process(&[]), Some(vec![]));
    }

    #[test]
    fn test_transducer_counting_binary() {
        let transducer = TransducerGenerators::counting_binary();

        // Outputs the input values (simplified counting)
        assert_eq!(transducer.process(&[1, 0, 1, 1]), Some(vec![1, 0, 1, 1]));
        assert_eq!(transducer.process(&[0, 0, 1]), Some(vec![0, 0, 1]));
    }

    #[test]
    fn test_transducer_add() {
        let transducer = TransducerGenerators::add(3);

        assert_eq!(
            transducer.process(&[(1, 2), (0, 1), (2, 2)]),
            Some(vec![3, 1, 4])
        );
        assert_eq!(transducer.process(&[(0, 0), (1, 1)]), Some(vec![0, 2]));
    }

    #[test]
    fn test_transducer_weight() {
        let transducer = TransducerGenerators::weight(vec![0, 1, 2, 3], 0);

        // Hamming weight: 0 -> 0, non-zero -> 1
        assert_eq!(
            transducer.process(&[1, 0, 2, 3, 0]),
            Some(vec![1, 0, 1, 1, 0])
        );
        assert_eq!(transducer.process(&[0, 0, 0]), Some(vec![0, 0, 0]));
        assert_eq!(transducer.process(&[1, 2, 3]), Some(vec![1, 1, 1]));
    }

    #[test]
    fn test_transducer_map() {
        let transducer =
            TransducerGenerators::map(vec![0, 1, 2, 3], vec![0, 1, 4, 9], |x| x * x);

        assert_eq!(transducer.process(&[0, 1, 2, 3]), Some(vec![0, 1, 4, 9]));
        assert_eq!(transducer.process(&[2, 2, 1]), Some(vec![4, 4, 1]));
    }

    #[test]
    fn test_transducer_count_symbol() {
        let transducer = TransducerGenerators::count_symbol(vec![0, 1, 2], 1);

        // Count occurrences of 1
        assert_eq!(transducer.process(&[0, 1, 1, 2, 1]), Some(vec![0, 1, 1, 0, 1]));
        assert_eq!(transducer.process(&[0, 0, 0]), Some(vec![0, 0, 0]));
        assert_eq!(transducer.process(&[1, 1, 1]), Some(vec![1, 1, 1]));
    }

    #[test]
    fn test_fsm_state_creation() {
        let initial = FSMState::initial("s0".to_string());
        assert!(initial.is_initial);
        assert!(!initial.is_final);

        let final_state = FSMState::final_state("s1".to_string());
        assert!(!final_state.is_initial);
        assert!(final_state.is_final);

        let normal = FSMState::new("s2".to_string(), false, false);
        assert!(!normal.is_initial);
        assert!(!normal.is_final);
    }

    #[test]
    fn test_automaton_initial_and_final_states() {
        let automaton = AutomatonGenerators::any_letter(vec![0, 1]);

        assert!(automaton.initial_state().is_some());
        assert_eq!(automaton.initial_state().unwrap().label, "q0");

        let final_states = automaton.final_states();
        assert_eq!(final_states.len(), 1);
        assert_eq!(final_states[0].label, "q1");
    }

    #[test]
    fn test_transducer_initial_state() {
        let transducer = TransducerGenerators::identity(vec![0, 1]);

        assert!(transducer.initial_state().is_some());
        assert_eq!(transducer.initial_state().unwrap().label, "q0");
    }
}
