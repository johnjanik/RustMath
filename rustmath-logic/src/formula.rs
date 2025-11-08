//! Boolean formulas and expressions

use std::collections::{HashMap, HashSet};
use std::fmt;

/// A variable in a Boolean formula
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Variable(pub String);

impl Variable {
    pub fn new(name: impl Into<String>) -> Self {
        Variable(name.into())
    }

    pub fn name(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A Boolean formula
#[derive(Clone, Debug, PartialEq)]
pub enum Formula {
    /// Constant true
    True,
    /// Constant false
    False,
    /// Variable
    Var(Variable),
    /// Negation
    Not(Box<Formula>),
    /// Conjunction (AND)
    And(Vec<Formula>),
    /// Disjunction (OR)
    Or(Vec<Formula>),
    /// Implication
    Implies(Box<Formula>, Box<Formula>),
    /// Biconditional (if and only if)
    Iff(Box<Formula>, Box<Formula>),
    /// Exclusive OR
    Xor(Box<Formula>, Box<Formula>),
}

impl Formula {
    /// Create a variable
    pub fn var(name: impl Into<String>) -> Self {
        Formula::Var(Variable::new(name))
    }

    /// Create a negation
    pub fn not(f: Formula) -> Self {
        Formula::Not(Box::new(f))
    }

    /// Create a conjunction
    pub fn and(formulas: Vec<Formula>) -> Self {
        Formula::And(formulas)
    }

    /// Create a disjunction
    pub fn or(formulas: Vec<Formula>) -> Self {
        Formula::Or(formulas)
    }

    /// Create an implication
    pub fn implies(left: Formula, right: Formula) -> Self {
        Formula::Implies(Box::new(left), Box::new(right))
    }

    /// Create a biconditional
    pub fn iff(left: Formula, right: Formula) -> Self {
        Formula::Iff(Box::new(left), Box::new(right))
    }

    /// Create an exclusive OR
    pub fn xor(left: Formula, right: Formula) -> Self {
        Formula::Xor(Box::new(left), Box::new(right))
    }

    /// Evaluate the formula given an assignment of variables
    pub fn evaluate(&self, assignment: &HashMap<Variable, bool>) -> Option<bool> {
        match self {
            Formula::True => Some(true),
            Formula::False => Some(false),
            Formula::Var(v) => assignment.get(v).copied(),
            Formula::Not(f) => f.evaluate(assignment).map(|b| !b),
            Formula::And(fs) => {
                let mut result = true;
                for f in fs {
                    match f.evaluate(assignment) {
                        Some(false) => return Some(false),
                        Some(true) => continue,
                        None => result = false, // Keep checking but mark as unknown
                    }
                }
                if result { Some(true) } else { None }
            }
            Formula::Or(fs) => {
                let mut all_known = true;
                for f in fs {
                    match f.evaluate(assignment) {
                        Some(true) => return Some(true),
                        Some(false) => continue,
                        None => all_known = false,
                    }
                }
                if all_known { Some(false) } else { None }
            }
            Formula::Implies(left, right) => {
                match (left.evaluate(assignment), right.evaluate(assignment)) {
                    (Some(false), _) => Some(true),
                    (Some(true), Some(r)) => Some(r),
                    (_, Some(true)) => Some(true),
                    _ => None,
                }
            }
            Formula::Iff(left, right) => {
                match (left.evaluate(assignment), right.evaluate(assignment)) {
                    (Some(l), Some(r)) => Some(l == r),
                    _ => None,
                }
            }
            Formula::Xor(left, right) => {
                match (left.evaluate(assignment), right.evaluate(assignment)) {
                    (Some(l), Some(r)) => Some(l != r),
                    _ => None,
                }
            }
        }
    }

    /// Get all variables in the formula
    pub fn variables(&self) -> HashSet<Variable> {
        let mut vars = HashSet::new();
        self.collect_variables(&mut vars);
        vars
    }

    fn collect_variables(&self, vars: &mut HashSet<Variable>) {
        match self {
            Formula::True | Formula::False => {}
            Formula::Var(v) => {
                vars.insert(v.clone());
            }
            Formula::Not(f) => f.collect_variables(vars),
            Formula::And(fs) | Formula::Or(fs) => {
                for f in fs {
                    f.collect_variables(vars);
                }
            }
            Formula::Implies(left, right) | Formula::Iff(left, right) | Formula::Xor(left, right) => {
                left.collect_variables(vars);
                right.collect_variables(vars);
            }
        }
    }

    /// Check if the formula is satisfiable (has a satisfying assignment)
    pub fn is_satisfiable(&self) -> bool {
        let vars: Vec<_> = self.variables().into_iter().collect();
        self.check_satisfiable(&vars, &HashMap::new(), 0)
    }

    fn check_satisfiable(
        &self,
        vars: &[Variable],
        assignment: &HashMap<Variable, bool>,
        idx: usize,
    ) -> bool {
        if idx >= vars.len() {
            return self.evaluate(assignment) == Some(true);
        }

        let var = &vars[idx];

        // Try assigning true
        let mut assignment_true = assignment.clone();
        assignment_true.insert(var.clone(), true);
        if self.check_satisfiable(vars, &assignment_true, idx + 1) {
            return true;
        }

        // Try assigning false
        let mut assignment_false = assignment.clone();
        assignment_false.insert(var.clone(), false);
        self.check_satisfiable(vars, &assignment_false, idx + 1)
    }

    /// Check if the formula is a tautology (always true)
    pub fn is_tautology(&self) -> bool {
        let vars: Vec<_> = self.variables().into_iter().collect();
        self.check_tautology(&vars, &HashMap::new(), 0)
    }

    fn check_tautology(
        &self,
        vars: &[Variable],
        assignment: &HashMap<Variable, bool>,
        idx: usize,
    ) -> bool {
        if idx >= vars.len() {
            return self.evaluate(assignment) == Some(true);
        }

        let var = &vars[idx];

        // Check both assignments
        let mut assignment_true = assignment.clone();
        assignment_true.insert(var.clone(), true);
        if !self.check_tautology(vars, &assignment_true, idx + 1) {
            return false;
        }

        let mut assignment_false = assignment.clone();
        assignment_false.insert(var.clone(), false);
        self.check_tautology(vars, &assignment_false, idx + 1)
    }

    /// Simplify the formula
    pub fn simplify(&self) -> Formula {
        match self {
            Formula::True | Formula::False | Formula::Var(_) => self.clone(),
            Formula::Not(f) => {
                let simplified = f.simplify();
                match simplified {
                    Formula::True => Formula::False,
                    Formula::False => Formula::True,
                    Formula::Not(inner) => (*inner).clone(),
                    _ => Formula::Not(Box::new(simplified)),
                }
            }
            Formula::And(fs) => {
                let mut simplified = Vec::new();
                for f in fs {
                    let s = f.simplify();
                    match s {
                        Formula::False => return Formula::False,
                        Formula::True => continue,
                        Formula::And(inner) => simplified.extend(inner),
                        _ => simplified.push(s),
                    }
                }
                match simplified.len() {
                    0 => Formula::True,
                    1 => simplified.into_iter().next().unwrap(),
                    _ => Formula::And(simplified),
                }
            }
            Formula::Or(fs) => {
                let mut simplified = Vec::new();
                for f in fs {
                    let s = f.simplify();
                    match s {
                        Formula::True => return Formula::True,
                        Formula::False => continue,
                        Formula::Or(inner) => simplified.extend(inner),
                        _ => simplified.push(s),
                    }
                }
                match simplified.len() {
                    0 => Formula::False,
                    1 => simplified.into_iter().next().unwrap(),
                    _ => Formula::Or(simplified),
                }
            }
            Formula::Implies(left, right) => {
                let l = left.simplify();
                let r = right.simplify();
                match (&l, &r) {
                    (Formula::False, _) => Formula::True,
                    (_, Formula::True) => Formula::True,
                    (Formula::True, _) => r,
                    _ => Formula::Implies(Box::new(l), Box::new(r)),
                }
            }
            Formula::Iff(left, right) => {
                let l = left.simplify();
                let r = right.simplify();
                match (&l, &r) {
                    (Formula::True, _) => r,
                    (_, Formula::True) => l,
                    (Formula::False, f) | (f, Formula::False) => Formula::Not(Box::new(f.clone())),
                    _ => Formula::Iff(Box::new(l), Box::new(r)),
                }
            }
            Formula::Xor(left, right) => {
                let l = left.simplify();
                let r = right.simplify();
                match (&l, &r) {
                    (Formula::True, f) | (f, Formula::True) => Formula::Not(Box::new(f.clone())),
                    (Formula::False, _) => r,
                    (_, Formula::False) => l,
                    _ => Formula::Xor(Box::new(l), Box::new(r)),
                }
            }
        }
    }
}

impl fmt::Display for Formula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Formula::True => write!(f, "⊤"),
            Formula::False => write!(f, "⊥"),
            Formula::Var(v) => write!(f, "{}", v),
            Formula::Not(inner) => write!(f, "¬{}", inner),
            Formula::And(fs) => {
                write!(f, "(")?;
                for (i, formula) in fs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ∧ ")?;
                    }
                    write!(f, "{}", formula)?;
                }
                write!(f, ")")
            }
            Formula::Or(fs) => {
                write!(f, "(")?;
                for (i, formula) in fs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ∨ ")?;
                    }
                    write!(f, "{}", formula)?;
                }
                write!(f, ")")
            }
            Formula::Implies(left, right) => write!(f, "({} → {})", left, right),
            Formula::Iff(left, right) => write!(f, "({} ↔ {})", left, right),
            Formula::Xor(left, right) => write!(f, "({} ⊕ {})", left, right),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_simple() {
        let f = Formula::var("x");
        let mut assignment = HashMap::new();
        assignment.insert(Variable::new("x"), true);
        assert_eq!(f.evaluate(&assignment), Some(true));

        assignment.insert(Variable::new("x"), false);
        assert_eq!(f.evaluate(&assignment), Some(false));
    }

    #[test]
    fn test_evaluate_and() {
        let f = Formula::and(vec![Formula::var("x"), Formula::var("y")]);

        let mut assignment = HashMap::new();
        assignment.insert(Variable::new("x"), true);
        assignment.insert(Variable::new("y"), true);
        assert_eq!(f.evaluate(&assignment), Some(true));

        assignment.insert(Variable::new("y"), false);
        assert_eq!(f.evaluate(&assignment), Some(false));
    }

    #[test]
    fn test_evaluate_or() {
        let f = Formula::or(vec![Formula::var("x"), Formula::var("y")]);

        let mut assignment = HashMap::new();
        assignment.insert(Variable::new("x"), true);
        assignment.insert(Variable::new("y"), false);
        assert_eq!(f.evaluate(&assignment), Some(true));

        assignment.insert(Variable::new("x"), false);
        assert_eq!(f.evaluate(&assignment), Some(false));
    }

    #[test]
    fn test_evaluate_not() {
        let f = Formula::not(Formula::var("x"));

        let mut assignment = HashMap::new();
        assignment.insert(Variable::new("x"), true);
        assert_eq!(f.evaluate(&assignment), Some(false));

        assignment.insert(Variable::new("x"), false);
        assert_eq!(f.evaluate(&assignment), Some(true));
    }

    #[test]
    fn test_evaluate_implies() {
        let f = Formula::implies(Formula::var("x"), Formula::var("y"));

        let mut assignment = HashMap::new();
        assignment.insert(Variable::new("x"), true);
        assignment.insert(Variable::new("y"), true);
        assert_eq!(f.evaluate(&assignment), Some(true));

        assignment.insert(Variable::new("y"), false);
        assert_eq!(f.evaluate(&assignment), Some(false));

        assignment.insert(Variable::new("x"), false);
        assert_eq!(f.evaluate(&assignment), Some(true));
    }

    #[test]
    fn test_variables() {
        let f = Formula::and(vec![
            Formula::var("x"),
            Formula::or(vec![Formula::var("y"), Formula::var("z")]),
        ]);

        let vars = f.variables();
        assert_eq!(vars.len(), 3);
        assert!(vars.contains(&Variable::new("x")));
        assert!(vars.contains(&Variable::new("y")));
        assert!(vars.contains(&Variable::new("z")));
    }

    #[test]
    fn test_is_satisfiable() {
        // x ∧ y is satisfiable
        let f = Formula::and(vec![Formula::var("x"), Formula::var("y")]);
        assert!(f.is_satisfiable());

        // x ∧ ¬x is not satisfiable
        let f = Formula::and(vec![Formula::var("x"), Formula::not(Formula::var("x"))]);
        assert!(!f.is_satisfiable());
    }

    #[test]
    fn test_is_tautology() {
        // x ∨ ¬x is a tautology
        let f = Formula::or(vec![Formula::var("x"), Formula::not(Formula::var("x"))]);
        assert!(f.is_tautology());

        // x ∧ y is not a tautology
        let f = Formula::and(vec![Formula::var("x"), Formula::var("y")]);
        assert!(!f.is_tautology());
    }

    #[test]
    fn test_simplify_double_negation() {
        let f = Formula::not(Formula::not(Formula::var("x")));
        let simplified = f.simplify();
        assert_eq!(simplified, Formula::var("x"));
    }

    #[test]
    fn test_simplify_and() {
        // x ∧ true = x
        let f = Formula::and(vec![Formula::var("x"), Formula::True]);
        assert_eq!(f.simplify(), Formula::var("x"));

        // x ∧ false = false
        let f = Formula::and(vec![Formula::var("x"), Formula::False]);
        assert_eq!(f.simplify(), Formula::False);
    }

    #[test]
    fn test_simplify_or() {
        // x ∨ false = x
        let f = Formula::or(vec![Formula::var("x"), Formula::False]);
        assert_eq!(f.simplify(), Formula::var("x"));

        // x ∨ true = true
        let f = Formula::or(vec![Formula::var("x"), Formula::True]);
        assert_eq!(f.simplify(), Formula::True);
    }
}
