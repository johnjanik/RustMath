//! Finite State Machines and Automata
//!
//! This crate provides implementations of various types of finite automata:
//! - DFA (Deterministic Finite Automaton)
//! - NFA (Non-deterministic Finite Automaton)
//! - Moore machines (output depends on state)
//! - Mealy machines (output depends on state and input)
//!
//! It also includes algorithms for:
//! - NFA to DFA conversion (subset construction)
//! - DFA minimization (Hopcroft's algorithm)
//! - Automata composition and operations

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Display;
use std::hash::Hash;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AutomataError {
    #[error("Invalid state: {0}")]
    InvalidState(String),
    #[error("Invalid transition")]
    InvalidTransition,
    #[error("No accepting state reached")]
    NotAccepted,
    #[error("Undefined transition from state {state} on input {input}")]
    UndefinedTransition { state: String, input: String },
}

/// Deterministic Finite Automaton (DFA)
///
/// A DFA is defined by (Q, Σ, δ, q₀, F) where:
/// - Q is a finite set of states
/// - Σ is a finite alphabet
/// - δ: Q × Σ → Q is the transition function
/// - q₀ ∈ Q is the initial state
/// - F ⊆ Q is the set of accepting states
#[derive(Debug, Clone)]
pub struct DFA<S, I>
where
    S: Clone + Eq + Hash,
    I: Clone + Eq + Hash,
{
    states: HashSet<S>,
    alphabet: HashSet<I>,
    transitions: HashMap<(S, I), S>,
    initial_state: S,
    accepting_states: HashSet<S>,
}

impl<S, I> DFA<S, I>
where
    S: Clone + Eq + Hash,
    I: Clone + Eq + Hash,
{
    /// Create a new DFA
    pub fn new(
        states: HashSet<S>,
        alphabet: HashSet<I>,
        transitions: HashMap<(S, I), S>,
        initial_state: S,
        accepting_states: HashSet<S>,
    ) -> Result<Self, AutomataError> {
        // Validate that initial state is in states
        if !states.contains(&initial_state) {
            return Err(AutomataError::InvalidState(
                "Initial state not in state set".to_string(),
            ));
        }

        // Validate that accepting states are in states
        for state in &accepting_states {
            if !states.contains(state) {
                return Err(AutomataError::InvalidState(
                    "Accepting state not in state set".to_string(),
                ));
            }
        }

        Ok(DFA {
            states,
            alphabet,
            transitions,
            initial_state,
            accepting_states,
        })
    }

    /// Run the DFA on an input string and return whether it's accepted
    pub fn accepts(&self, input: &[I]) -> bool {
        match self.run(input) {
            Ok(state) => self.accepting_states.contains(&state),
            Err(_) => false,
        }
    }

    /// Run the DFA on input and return the final state
    pub fn run(&self, input: &[I]) -> Result<S, AutomataError> {
        let mut current_state = self.initial_state.clone();

        for symbol in input {
            if !self.alphabet.contains(symbol) {
                return Err(AutomataError::InvalidTransition);
            }

            current_state = self
                .transitions
                .get(&(current_state.clone(), symbol.clone()))
                .ok_or(AutomataError::InvalidTransition)?
                .clone();
        }

        Ok(current_state)
    }

    /// Get the set of states
    pub fn states(&self) -> &HashSet<S> {
        &self.states
    }

    /// Get the alphabet
    pub fn alphabet(&self) -> &HashSet<I> {
        &self.alphabet
    }

    /// Get the initial state
    pub fn initial_state(&self) -> &S {
        &self.initial_state
    }

    /// Get the accepting states
    pub fn accepting_states(&self) -> &HashSet<S> {
        &self.accepting_states
    }

    /// Get the transition function
    pub fn transitions(&self) -> &HashMap<(S, I), S> {
        &self.transitions
    }

    /// Minimize the DFA using Hopcroft's algorithm
    pub fn minimize(&self) -> DFA<usize, I>
    where
        S: Ord + Display,
    {
        // Create a mapping from states to indices
        let mut state_vec: Vec<S> = self.states.iter().cloned().collect();
        state_vec.sort();
        let state_to_idx: HashMap<S, usize> = state_vec
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        // Initialize partitions: accepting and non-accepting states
        let mut partitions: Vec<HashSet<usize>> = vec![HashSet::new(), HashSet::new()];

        for (i, state) in state_vec.iter().enumerate() {
            if self.accepting_states.contains(state) {
                partitions[0].insert(i);
            } else {
                partitions[1].insert(i);
            }
        }

        // Remove empty partitions
        partitions.retain(|p| !p.is_empty());

        // Refine partitions
        let mut changed = true;
        while changed {
            changed = false;
            let mut new_partitions = Vec::new();

            for partition in &partitions {
                let mut splits: HashMap<Vec<Option<usize>>, HashSet<usize>> = HashMap::new();

                for &state_idx in partition {
                    let state = &state_vec[state_idx];
                    let mut signature = Vec::new();

                    for symbol in &self.alphabet {
                        if let Some(next_state) =
                            self.transitions.get(&(state.clone(), symbol.clone()))
                        {
                            let next_idx = state_to_idx[next_state];
                            // Find which partition next_idx belongs to
                            let part_idx = partitions
                                .iter()
                                .position(|p| p.contains(&next_idx))
                                .unwrap();
                            signature.push(Some(part_idx));
                        } else {
                            signature.push(None);
                        }
                    }

                    splits.entry(signature).or_insert_with(HashSet::new).insert(state_idx);
                }

                if splits.len() > 1 {
                    changed = true;
                }

                new_partitions.extend(splits.into_values());
            }

            partitions = new_partitions;
        }

        // Build minimized DFA
        let partition_of: HashMap<usize, usize> = partitions
            .iter()
            .enumerate()
            .flat_map(|(part_idx, part)| part.iter().map(move |&state_idx| (state_idx, part_idx)))
            .collect();

        let mut min_states = HashSet::new();
        let mut min_transitions = HashMap::new();
        let mut min_accepting = HashSet::new();

        for (part_idx, _) in partitions.iter().enumerate() {
            min_states.insert(part_idx);
        }

        let initial_idx = state_to_idx[&self.initial_state];
        let min_initial = partition_of[&initial_idx];

        // Build transitions for minimized DFA
        for (part_idx, partition) in partitions.iter().enumerate() {
            let representative = *partition.iter().next().unwrap();
            let rep_state = &state_vec[representative];

            if self.accepting_states.contains(rep_state) {
                min_accepting.insert(part_idx);
            }

            for symbol in &self.alphabet {
                if let Some(next_state) = self.transitions.get(&(rep_state.clone(), symbol.clone()))
                {
                    let next_idx = state_to_idx[next_state];
                    let next_part = partition_of[&next_idx];
                    min_transitions.insert((part_idx, symbol.clone()), next_part);
                }
            }
        }

        DFA::new(
            min_states,
            self.alphabet.clone(),
            min_transitions,
            min_initial,
            min_accepting,
        )
        .unwrap()
    }

    /// Complement the DFA (accepts the complement language)
    pub fn complement(&self) -> Self {
        let new_accepting: HashSet<S> = self
            .states
            .iter()
            .filter(|s| !self.accepting_states.contains(s))
            .cloned()
            .collect();

        DFA::new(
            self.states.clone(),
            self.alphabet.clone(),
            self.transitions.clone(),
            self.initial_state.clone(),
            new_accepting,
        )
        .unwrap()
    }
}

/// Non-deterministic Finite Automaton (NFA)
///
/// An NFA is defined by (Q, Σ, δ, q₀, F) where:
/// - Q is a finite set of states
/// - Σ is a finite alphabet
/// - δ: Q × (Σ ∪ {ε}) → P(Q) is the transition function
/// - q₀ ∈ Q is the initial state
/// - F ⊆ Q is the set of accepting states
#[derive(Debug, Clone)]
pub struct NFA<S, I>
where
    S: Clone + Eq + Hash,
    I: Clone + Eq + Hash,
{
    states: HashSet<S>,
    alphabet: HashSet<I>,
    transitions: HashMap<(S, Option<I>), HashSet<S>>, // Option<I> for epsilon transitions
    initial_state: S,
    accepting_states: HashSet<S>,
}

impl<S, I> NFA<S, I>
where
    S: Clone + Eq + Hash,
    I: Clone + Eq + Hash,
{
    /// Create a new NFA
    pub fn new(
        states: HashSet<S>,
        alphabet: HashSet<I>,
        transitions: HashMap<(S, Option<I>), HashSet<S>>,
        initial_state: S,
        accepting_states: HashSet<S>,
    ) -> Result<Self, AutomataError> {
        if !states.contains(&initial_state) {
            return Err(AutomataError::InvalidState(
                "Initial state not in state set".to_string(),
            ));
        }

        for state in &accepting_states {
            if !states.contains(state) {
                return Err(AutomataError::InvalidState(
                    "Accepting state not in state set".to_string(),
                ));
            }
        }

        Ok(NFA {
            states,
            alphabet,
            transitions,
            initial_state,
            accepting_states,
        })
    }

    /// Compute epsilon closure of a set of states
    pub fn epsilon_closure(&self, states: &HashSet<S>) -> HashSet<S> {
        let mut closure = states.clone();
        let mut stack: Vec<S> = states.iter().cloned().collect();

        while let Some(state) = stack.pop() {
            if let Some(epsilon_transitions) = self.transitions.get(&(state, None)) {
                for next_state in epsilon_transitions {
                    if !closure.contains(next_state) {
                        closure.insert(next_state.clone());
                        stack.push(next_state.clone());
                    }
                }
            }
        }

        closure
    }

    /// Run the NFA on an input string and return whether it's accepted
    pub fn accepts(&self, input: &[I]) -> bool {
        let mut current_states = HashSet::new();
        current_states.insert(self.initial_state.clone());
        current_states = self.epsilon_closure(&current_states);

        for symbol in input {
            let mut next_states = HashSet::new();

            for state in &current_states {
                if let Some(transitions) = self.transitions.get(&(state.clone(), Some(symbol.clone())))
                {
                    next_states.extend(transitions.iter().cloned());
                }
            }

            if next_states.is_empty() {
                return false;
            }

            current_states = self.epsilon_closure(&next_states);
        }

        current_states
            .iter()
            .any(|s| self.accepting_states.contains(s))
    }

    /// Convert NFA to DFA using subset construction
    /// Returns a DFA where each state is a sorted vector of NFA states
    pub fn to_dfa(&self) -> DFA<Vec<S>, I>
    where
        S: Ord,
    {
        let mut dfa_states = HashSet::new();
        let mut dfa_transitions = HashMap::new();
        let mut dfa_accepting = HashSet::new();

        // Convert to sorted vec for hashing
        let to_sorted_vec = |set: &HashSet<S>| -> Vec<S> {
            let mut v: Vec<S> = set.iter().cloned().collect();
            v.sort();
            v
        };

        let mut initial_set = HashSet::new();
        initial_set.insert(self.initial_state.clone());
        let initial_closure = self.epsilon_closure(&initial_set);
        let dfa_initial = to_sorted_vec(&initial_closure);

        let mut worklist = VecDeque::new();
        worklist.push_back(dfa_initial.clone());
        dfa_states.insert(dfa_initial.clone());

        while let Some(state_vec) = worklist.pop_front() {
            // Convert back to HashSet for processing
            let state_set: HashSet<S> = state_vec.iter().cloned().collect();

            // Check if this is an accepting state
            if state_set.iter().any(|s| self.accepting_states.contains(s)) {
                dfa_accepting.insert(state_vec.clone());
            }

            // For each symbol, compute the next state set
            for symbol in &self.alphabet {
                let mut next_states = HashSet::new();

                for state in &state_set {
                    if let Some(transitions) =
                        self.transitions.get(&(state.clone(), Some(symbol.clone())))
                    {
                        next_states.extend(transitions.iter().cloned());
                    }
                }

                if !next_states.is_empty() {
                    let next_closure = self.epsilon_closure(&next_states);
                    let next_state_vec = to_sorted_vec(&next_closure);

                    if !dfa_states.contains(&next_state_vec) {
                        dfa_states.insert(next_state_vec.clone());
                        worklist.push_back(next_state_vec.clone());
                    }

                    dfa_transitions.insert((state_vec.clone(), symbol.clone()), next_state_vec);
                }
            }
        }

        DFA::new(
            dfa_states,
            self.alphabet.clone(),
            dfa_transitions,
            dfa_initial,
            dfa_accepting,
        )
        .unwrap()
    }

    /// Get the set of states
    pub fn states(&self) -> &HashSet<S> {
        &self.states
    }

    /// Get the alphabet
    pub fn alphabet(&self) -> &HashSet<I> {
        &self.alphabet
    }

    /// Get the initial state
    pub fn initial_state(&self) -> &S {
        &self.initial_state
    }

    /// Get the accepting states
    pub fn accepting_states(&self) -> &HashSet<S> {
        &self.accepting_states
    }
}

/// Moore Machine
///
/// A Moore machine is a finite state machine where outputs depend only on the state.
/// It is defined by (Q, Σ, Ω, δ, λ, q₀) where:
/// - Q is a finite set of states
/// - Σ is the input alphabet
/// - Ω is the output alphabet
/// - δ: Q × Σ → Q is the transition function
/// - λ: Q → Ω is the output function
/// - q₀ ∈ Q is the initial state
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MooreMachine<S, I, O>
where
    S: Clone + Eq + Hash,
    I: Clone + Eq + Hash,
    O: Clone,
{
    states: HashSet<S>,
    input_alphabet: HashSet<I>,
    transitions: HashMap<(S, I), S>,
    output_function: HashMap<S, O>,
    initial_state: S,
}

impl<S, I, O> MooreMachine<S, I, O>
where
    S: Clone + Eq + Hash,
    I: Clone + Eq + Hash,
    O: Clone,
{
    /// Create a new Moore machine
    pub fn new(
        states: HashSet<S>,
        input_alphabet: HashSet<I>,
        transitions: HashMap<(S, I), S>,
        output_function: HashMap<S, O>,
        initial_state: S,
    ) -> Result<Self, AutomataError> {
        if !states.contains(&initial_state) {
            return Err(AutomataError::InvalidState(
                "Initial state not in state set".to_string(),
            ));
        }

        Ok(MooreMachine {
            states,
            input_alphabet,
            transitions,
            output_function,
            initial_state,
        })
    }

    /// Run the Moore machine on input and return the sequence of outputs
    pub fn run(&self, input: &[I]) -> Result<Vec<O>, AutomataError> {
        let mut current_state = self.initial_state.clone();
        let mut outputs = Vec::new();

        // Output for initial state
        if let Some(output) = self.output_function.get(&current_state) {
            outputs.push(output.clone());
        }

        for symbol in input {
            current_state = self
                .transitions
                .get(&(current_state.clone(), symbol.clone()))
                .ok_or(AutomataError::InvalidTransition)?
                .clone();

            if let Some(output) = self.output_function.get(&current_state) {
                outputs.push(output.clone());
            }
        }

        Ok(outputs)
    }

    /// Get the output for a specific state
    pub fn output(&self, state: &S) -> Option<&O> {
        self.output_function.get(state)
    }

    /// Get the current state after processing input
    pub fn state_after(&self, input: &[I]) -> Result<S, AutomataError> {
        let mut current_state = self.initial_state.clone();

        for symbol in input {
            current_state = self
                .transitions
                .get(&(current_state.clone(), symbol.clone()))
                .ok_or(AutomataError::InvalidTransition)?
                .clone();
        }

        Ok(current_state)
    }
}

/// Mealy Machine
///
/// A Mealy machine is a finite state machine where outputs depend on both state and input.
/// It is defined by (Q, Σ, Ω, δ, λ, q₀) where:
/// - Q is a finite set of states
/// - Σ is the input alphabet
/// - Ω is the output alphabet
/// - δ: Q × Σ → Q is the transition function
/// - λ: Q × Σ → Ω is the output function
/// - q₀ ∈ Q is the initial state
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MealyMachine<S, I, O>
where
    S: Clone + Eq + Hash,
    I: Clone + Eq + Hash,
    O: Clone,
{
    states: HashSet<S>,
    input_alphabet: HashSet<I>,
    transitions: HashMap<(S, I), S>,
    output_function: HashMap<(S, I), O>,
    initial_state: S,
}

impl<S, I, O> MealyMachine<S, I, O>
where
    S: Clone + Eq + Hash,
    I: Clone + Eq + Hash,
    O: Clone,
{
    /// Create a new Mealy machine
    pub fn new(
        states: HashSet<S>,
        input_alphabet: HashSet<I>,
        transitions: HashMap<(S, I), S>,
        output_function: HashMap<(S, I), O>,
        initial_state: S,
    ) -> Result<Self, AutomataError> {
        if !states.contains(&initial_state) {
            return Err(AutomataError::InvalidState(
                "Initial state not in state set".to_string(),
            ));
        }

        Ok(MealyMachine {
            states,
            input_alphabet,
            transitions,
            output_function,
            initial_state,
        })
    }

    /// Run the Mealy machine on input and return the sequence of outputs
    pub fn run(&self, input: &[I]) -> Result<Vec<O>, AutomataError> {
        let mut current_state = self.initial_state.clone();
        let mut outputs = Vec::new();

        for symbol in input {
            if let Some(output) = self.output_function.get(&(current_state.clone(), symbol.clone()))
            {
                outputs.push(output.clone());
            }

            current_state = self
                .transitions
                .get(&(current_state.clone(), symbol.clone()))
                .ok_or(AutomataError::InvalidTransition)?
                .clone();
        }

        Ok(outputs)
    }

    /// Get the output for a specific state and input
    pub fn output(&self, state: &S, input: &I) -> Option<&O> {
        self.output_function.get(&(state.clone(), input.clone()))
    }

    /// Convert Mealy machine to Moore machine
    pub fn to_moore(&self) -> MooreMachine<(S, Option<O>), I, O>
    where
        O: Eq + Hash,
    {
        // Create new states that include the output
        let mut moore_states = HashSet::new();
        let mut moore_transitions = HashMap::new();
        let mut moore_outputs = HashMap::new();

        // Initial state has no previous output
        moore_states.insert((self.initial_state.clone(), None));

        let mut worklist = VecDeque::new();
        worklist.push_back((self.initial_state.clone(), None));

        while let Some(current_moore_state) = worklist.pop_front() {
            let (state, prev_output) = current_moore_state;

            for input in &self.input_alphabet {
                if let Some(next_state) = self.transitions.get(&(state.clone(), input.clone())) {
                    let output = self.output_function.get(&(state.clone(), input.clone()));

                    let new_moore_state = (next_state.clone(), output.cloned());

                    if !moore_states.contains(&new_moore_state) {
                        moore_states.insert(new_moore_state.clone());
                        worklist.push_back(new_moore_state.clone());
                    }

                    moore_transitions.insert(
                        ((state.clone(), prev_output.clone()), input.clone()),
                        new_moore_state.clone(),
                    );

                    if let Some(out) = output {
                        moore_outputs.insert(new_moore_state.clone(), out.clone());
                    }
                }
            }
        }

        MooreMachine::new(
            moore_states,
            self.input_alphabet.clone(),
            moore_transitions,
            moore_outputs,
            (self.initial_state.clone(), None),
        )
        .unwrap()
    }
}

/// Product construction for DFA intersection
pub fn dfa_intersection<S1, S2, I>(dfa1: &DFA<S1, I>, dfa2: &DFA<S2, I>) -> DFA<(S1, S2), I>
where
    S1: Clone + Eq + Hash,
    S2: Clone + Eq + Hash,
    I: Clone + Eq + Hash,
{
    let mut states = HashSet::new();
    let mut transitions = HashMap::new();
    let mut accepting = HashSet::new();

    // Product construction
    for s1 in dfa1.states() {
        for s2 in dfa2.states() {
            states.insert((s1.clone(), s2.clone()));

            if dfa1.accepting_states().contains(s1) && dfa2.accepting_states().contains(s2) {
                accepting.insert((s1.clone(), s2.clone()));
            }
        }
    }

    // Build transitions
    for s1 in dfa1.states() {
        for s2 in dfa2.states() {
            for symbol in dfa1.alphabet() {
                if dfa2.alphabet().contains(symbol) {
                    if let Some(next1) = dfa1.transitions().get(&(s1.clone(), symbol.clone())) {
                        if let Some(next2) = dfa2.transitions().get(&(s2.clone(), symbol.clone())) {
                            transitions.insert(
                                ((s1.clone(), s2.clone()), symbol.clone()),
                                (next1.clone(), next2.clone()),
                            );
                        }
                    }
                }
            }
        }
    }

    let initial = (dfa1.initial_state().clone(), dfa2.initial_state().clone());

    // Use the common alphabet (intersection)
    let alphabet: HashSet<I> = dfa1
        .alphabet()
        .intersection(dfa2.alphabet())
        .cloned()
        .collect();

    DFA::new(states, alphabet, transitions, initial, accepting).unwrap()
}

/// Product construction for DFA union
pub fn dfa_union<S1, S2, I>(dfa1: &DFA<S1, I>, dfa2: &DFA<S2, I>) -> DFA<(S1, S2), I>
where
    S1: Clone + Eq + Hash,
    S2: Clone + Eq + Hash,
    I: Clone + Eq + Hash,
{
    let mut states = HashSet::new();
    let mut transitions = HashMap::new();
    let mut accepting = HashSet::new();

    for s1 in dfa1.states() {
        for s2 in dfa2.states() {
            states.insert((s1.clone(), s2.clone()));

            if dfa1.accepting_states().contains(s1) || dfa2.accepting_states().contains(s2) {
                accepting.insert((s1.clone(), s2.clone()));
            }
        }
    }

    for s1 in dfa1.states() {
        for s2 in dfa2.states() {
            for symbol in dfa1.alphabet() {
                if dfa2.alphabet().contains(symbol) {
                    if let Some(next1) = dfa1.transitions().get(&(s1.clone(), symbol.clone())) {
                        if let Some(next2) = dfa2.transitions().get(&(s2.clone(), symbol.clone())) {
                            transitions.insert(
                                ((s1.clone(), s2.clone()), symbol.clone()),
                                (next1.clone(), next2.clone()),
                            );
                        }
                    }
                }
            }
        }
    }

    let initial = (dfa1.initial_state().clone(), dfa2.initial_state().clone());
    let alphabet: HashSet<I> = dfa1
        .alphabet()
        .intersection(dfa2.alphabet())
        .cloned()
        .collect();

    DFA::new(states, alphabet, transitions, initial, accepting).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dfa_simple() {
        // DFA that accepts strings ending with '1'
        let mut states = HashSet::new();
        states.insert("q0");
        states.insert("q1");

        let mut alphabet = HashSet::new();
        alphabet.insert('0');
        alphabet.insert('1');

        let mut transitions = HashMap::new();
        transitions.insert(("q0", '0'), "q0");
        transitions.insert(("q0", '1'), "q1");
        transitions.insert(("q1", '0'), "q0");
        transitions.insert(("q1", '1'), "q1");

        let mut accepting = HashSet::new();
        accepting.insert("q1");

        let dfa = DFA::new(states, alphabet, transitions, "q0", accepting).unwrap();

        assert!(dfa.accepts(&['1']));
        assert!(dfa.accepts(&['0', '1']));
        assert!(dfa.accepts(&['1', '0', '1']));
        assert!(!dfa.accepts(&['0']));
        assert!(!dfa.accepts(&['1', '0']));
    }

    #[test]
    fn test_dfa_complement() {
        let mut states = HashSet::new();
        states.insert("q0");
        states.insert("q1");

        let mut alphabet = HashSet::new();
        alphabet.insert('0');
        alphabet.insert('1');

        let mut transitions = HashMap::new();
        transitions.insert(("q0", '0'), "q0");
        transitions.insert(("q0", '1'), "q1");
        transitions.insert(("q1", '0'), "q0");
        transitions.insert(("q1", '1'), "q1");

        let mut accepting = HashSet::new();
        accepting.insert("q1");

        let dfa = DFA::new(states, alphabet, transitions, "q0", accepting).unwrap();
        let complement = dfa.complement();

        // Original accepts strings ending with '1', complement accepts strings NOT ending with '1'
        assert!(!complement.accepts(&['1']));
        assert!(!complement.accepts(&['0', '1']));
        assert!(complement.accepts(&['0']));
        assert!(complement.accepts(&['1', '0']));
    }

    #[test]
    fn test_nfa_simple() {
        // NFA that accepts strings containing "01"
        let mut states = HashSet::new();
        states.insert(0);
        states.insert(1);
        states.insert(2);

        let mut alphabet = HashSet::new();
        alphabet.insert('0');
        alphabet.insert('1');

        let mut transitions = HashMap::new();

        // From state 0
        let mut t0_0 = HashSet::new();
        t0_0.insert(0);
        t0_0.insert(1);
        transitions.insert((0, Some('0')), t0_0);

        let mut t0_1 = HashSet::new();
        t0_1.insert(0);
        transitions.insert((0, Some('1')), t0_1);

        // From state 1
        let mut t1_1 = HashSet::new();
        t1_1.insert(2);
        transitions.insert((1, Some('1')), t1_1);

        // From state 2 (accepting state - loop)
        let mut t2_0 = HashSet::new();
        t2_0.insert(2);
        transitions.insert((2, Some('0')), t2_0);

        let mut t2_1 = HashSet::new();
        t2_1.insert(2);
        transitions.insert((2, Some('1')), t2_1);

        let mut accepting = HashSet::new();
        accepting.insert(2);

        let nfa = NFA::new(states, alphabet, transitions, 0, accepting).unwrap();

        assert!(nfa.accepts(&['0', '1']));
        assert!(nfa.accepts(&['1', '0', '1']));
        assert!(nfa.accepts(&['0', '1', '0', '0']));
        assert!(!nfa.accepts(&['0']));
        assert!(!nfa.accepts(&['1']));
        assert!(!nfa.accepts(&['1', '0']));
    }

    #[test]
    fn test_nfa_with_epsilon() {
        // NFA with epsilon transitions
        let mut states = HashSet::new();
        states.insert(0);
        states.insert(1);
        states.insert(2);

        let mut alphabet = HashSet::new();
        alphabet.insert('a');
        alphabet.insert('b');

        let mut transitions = HashMap::new();

        // Epsilon transition from 0 to 1
        let mut eps = HashSet::new();
        eps.insert(1);
        transitions.insert((0, None), eps);

        // From state 0
        let mut t0_a = HashSet::new();
        t0_a.insert(0);
        transitions.insert((0, Some('a')), t0_a);

        // From state 1
        let mut t1_b = HashSet::new();
        t1_b.insert(2);
        transitions.insert((1, Some('b')), t1_b);

        let mut accepting = HashSet::new();
        accepting.insert(2);

        let nfa = NFA::new(states, alphabet, transitions, 0, accepting).unwrap();

        // Should accept 'b' because of epsilon transition to state 1
        assert!(nfa.accepts(&['b']));
        assert!(nfa.accepts(&['a', 'b']));
        assert!(nfa.accepts(&['a', 'a', 'b']));
        assert!(!nfa.accepts(&['a']));
    }

    #[test]
    fn test_nfa_to_dfa() {
        // Simple NFA
        let mut states = HashSet::new();
        states.insert(0);
        states.insert(1);

        let mut alphabet = HashSet::new();
        alphabet.insert('a');
        alphabet.insert('b');

        let mut transitions = HashMap::new();

        let mut t0_a = HashSet::new();
        t0_a.insert(0);
        t0_a.insert(1);
        transitions.insert((0, Some('a')), t0_a);

        let mut t0_b = HashSet::new();
        t0_b.insert(0);
        transitions.insert((0, Some('b')), t0_b);

        let mut t1_b = HashSet::new();
        t1_b.insert(1);
        transitions.insert((1, Some('b')), t1_b);

        let mut accepting = HashSet::new();
        accepting.insert(1);

        let nfa = NFA::new(states, alphabet, transitions, 0, accepting).unwrap();
        let dfa = nfa.to_dfa();

        // DFA should accept same language as NFA
        assert!(dfa.accepts(&['a']));
        assert!(dfa.accepts(&['a', 'b']));
        assert!(dfa.accepts(&['b', 'a']));
        assert!(!dfa.accepts(&['b']));
    }

    #[test]
    fn test_dfa_minimization() {
        // Create a DFA with redundant states
        let mut states = HashSet::new();
        states.insert("q0");
        states.insert("q1");
        states.insert("q2");
        states.insert("q3");

        let mut alphabet = HashSet::new();
        alphabet.insert('a');
        alphabet.insert('b');

        let mut transitions = HashMap::new();
        transitions.insert(("q0", 'a'), "q1");
        transitions.insert(("q0", 'b'), "q2");
        transitions.insert(("q1", 'a'), "q1");
        transitions.insert(("q1", 'b'), "q3");
        transitions.insert(("q2", 'a'), "q1");
        transitions.insert(("q2", 'b'), "q2");
        transitions.insert(("q3", 'a'), "q1");
        transitions.insert(("q3", 'b'), "q2");

        let mut accepting = HashSet::new();
        accepting.insert("q1");
        accepting.insert("q3");

        let dfa = DFA::new(states, alphabet, transitions, "q0", accepting).unwrap();
        let minimized = dfa.minimize();

        // Test that minimized DFA accepts same language
        let test_cases = vec![
            vec!['a'],
            vec!['b', 'a'],
            vec!['a', 'b'],
            vec!['a', 'a'],
            vec!['b'],
        ];

        for input in test_cases {
            assert_eq!(dfa.accepts(&input), minimized.accepts(&input));
        }

        // Minimized DFA should have fewer states
        assert!(minimized.states().len() <= dfa.states().len());
    }

    #[test]
    fn test_moore_machine() {
        // Simple Moore machine: outputs 0 for even count of 'a's, 1 for odd count
        let mut states = HashSet::new();
        states.insert("even");
        states.insert("odd");

        let mut alphabet = HashSet::new();
        alphabet.insert('a');

        let mut transitions = HashMap::new();
        transitions.insert(("even", 'a'), "odd");
        transitions.insert(("odd", 'a'), "even");

        let mut outputs = HashMap::new();
        outputs.insert("even", 0);
        outputs.insert("odd", 1);

        let moore = MooreMachine::new(states, alphabet, transitions, outputs, "even").unwrap();

        let result = moore.run(&['a', 'a', 'a']).unwrap();
        assert_eq!(result, vec![0, 1, 0, 1]); // Initial output + 3 transitions
    }

    #[test]
    fn test_mealy_machine() {
        // Mealy machine: outputs 1 when input is 'a', 0 when input is 'b'
        let mut states = HashSet::new();
        states.insert("q0");

        let mut alphabet = HashSet::new();
        alphabet.insert('a');
        alphabet.insert('b');

        let mut transitions = HashMap::new();
        transitions.insert(("q0", 'a'), "q0");
        transitions.insert(("q0", 'b'), "q0");

        let mut outputs = HashMap::new();
        outputs.insert(("q0", 'a'), 1);
        outputs.insert(("q0", 'b'), 0);

        let mealy = MealyMachine::new(states, alphabet, transitions, outputs, "q0").unwrap();

        let result = mealy.run(&['a', 'b', 'a', 'a']).unwrap();
        assert_eq!(result, vec![1, 0, 1, 1]);
    }

    #[test]
    fn test_dfa_intersection() {
        // DFA 1: accepts strings with even number of 'a's
        let mut states1 = HashSet::new();
        states1.insert("even");
        states1.insert("odd");

        let mut alphabet = HashSet::new();
        alphabet.insert('a');
        alphabet.insert('b');

        let mut trans1 = HashMap::new();
        trans1.insert(("even", 'a'), "odd");
        trans1.insert(("odd", 'a'), "even");
        trans1.insert(("even", 'b'), "even");
        trans1.insert(("odd", 'b'), "odd");

        let mut accept1 = HashSet::new();
        accept1.insert("even");

        let dfa1 = DFA::new(states1, alphabet.clone(), trans1, "even", accept1).unwrap();

        // DFA 2: accepts strings with even number of 'b's
        let mut states2 = HashSet::new();
        states2.insert("even");
        states2.insert("odd");

        let mut trans2 = HashMap::new();
        trans2.insert(("even", 'b'), "odd");
        trans2.insert(("odd", 'b'), "even");
        trans2.insert(("even", 'a'), "even");
        trans2.insert(("odd", 'a'), "odd");

        let mut accept2 = HashSet::new();
        accept2.insert("even");

        let dfa2 = DFA::new(states2, alphabet, trans2, "even", accept2).unwrap();

        let intersection = dfa_intersection(&dfa1, &dfa2);

        // Should accept strings with even number of both 'a's and 'b's
        assert!(intersection.accepts(&[]));
        assert!(intersection.accepts(&['a', 'a']));
        assert!(intersection.accepts(&['b', 'b']));
        assert!(intersection.accepts(&['a', 'a', 'b', 'b']));
        assert!(!intersection.accepts(&['a']));
        assert!(!intersection.accepts(&['b']));
        assert!(!intersection.accepts(&['a', 'b']));
    }

    #[test]
    fn test_dfa_union() {
        // DFA 1: accepts strings ending with 'a'
        let mut states1 = HashSet::new();
        states1.insert("q0");
        states1.insert("q1");

        let mut alphabet = HashSet::new();
        alphabet.insert('a');
        alphabet.insert('b');

        let mut trans1 = HashMap::new();
        trans1.insert(("q0", 'a'), "q1");
        trans1.insert(("q0", 'b'), "q0");
        trans1.insert(("q1", 'a'), "q1");
        trans1.insert(("q1", 'b'), "q0");

        let mut accept1 = HashSet::new();
        accept1.insert("q1");

        let dfa1 = DFA::new(states1, alphabet.clone(), trans1, "q0", accept1).unwrap();

        // DFA 2: accepts strings ending with 'b'
        let mut states2 = HashSet::new();
        states2.insert("p0");
        states2.insert("p1");

        let mut trans2 = HashMap::new();
        trans2.insert(("p0", 'b'), "p1");
        trans2.insert(("p0", 'a'), "p0");
        trans2.insert(("p1", 'b'), "p1");
        trans2.insert(("p1", 'a'), "p0");

        let mut accept2 = HashSet::new();
        accept2.insert("p1");

        let dfa2 = DFA::new(states2, alphabet, trans2, "p0", accept2).unwrap();

        let union = dfa_union(&dfa1, &dfa2);

        // Should accept strings ending with either 'a' or 'b'
        assert!(union.accepts(&['a']));
        assert!(union.accepts(&['b']));
        assert!(union.accepts(&['a', 'b']));
        assert!(union.accepts(&['b', 'a']));
        assert!(!union.accepts(&[]));
    }

    #[test]
    fn test_mealy_to_moore() {
        // Simple Mealy machine
        let mut states = HashSet::new();
        states.insert("q0");
        states.insert("q1");

        let mut alphabet = HashSet::new();
        alphabet.insert('a');
        alphabet.insert('b');

        let mut transitions = HashMap::new();
        transitions.insert(("q0", 'a'), "q1");
        transitions.insert(("q0", 'b'), "q0");
        transitions.insert(("q1", 'a'), "q1");
        transitions.insert(("q1", 'b'), "q0");

        let mut outputs = HashMap::new();
        outputs.insert(("q0", 'a'), "out1");
        outputs.insert(("q0", 'b'), "out0");
        outputs.insert(("q1", 'a'), "out1");
        outputs.insert(("q1", 'b'), "out0");

        let mealy =
            MealyMachine::new(states, alphabet, transitions, outputs, "q0").unwrap();

        let moore = mealy.to_moore();

        // Verify the conversion works by checking outputs match
        let mealy_output = mealy.run(&['a', 'b', 'a']).unwrap();
        let moore_output = moore.run(&['a', 'b', 'a']).unwrap();

        // The Moore machine's outputs after the initial state should match Mealy's outputs
        // (initial state has no output, then subsequent states have outputs)
        assert_eq!(mealy_output.len(), moore_output.len());
        assert_eq!(mealy_output, moore_output);
    }
}
