//! SAT solver using DPLL algorithm

use crate::cnf::{Clause, Cnf};
use crate::formula::Variable;
use std::collections::{HashMap, HashSet};

/// Result of SAT solving
#[derive(Clone, Debug, PartialEq)]
pub enum SatResult {
    Satisfiable(HashMap<Variable, bool>),
    Unsatisfiable,
}

impl SatResult {
    pub fn is_satisfiable(&self) -> bool {
        matches!(self, SatResult::Satisfiable(_))
    }

    pub fn is_unsatisfiable(&self) -> bool {
        matches!(self, SatResult::Unsatisfiable)
    }

    pub fn assignment(&self) -> Option<&HashMap<Variable, bool>> {
        match self {
            SatResult::Satisfiable(assignment) => Some(assignment),
            SatResult::Unsatisfiable => None,
        }
    }
}

/// SAT solver using the DPLL algorithm
pub struct SatSolver {
    cnf: Cnf,
}

impl SatSolver {
    pub fn new(cnf: Cnf) -> Self {
        SatSolver { cnf }
    }

    /// Solve the SAT problem
    pub fn solve(&self) -> SatResult {
        let assignment = HashMap::new();
        self.dpll(&self.cnf.clauses, &assignment)
    }

    /// DPLL algorithm
    fn dpll(&self, clauses: &[Clause], assignment: &HashMap<Variable, bool>) -> SatResult {
        // If all clauses are satisfied, return satisfiable
        if clauses.is_empty() {
            return SatResult::Satisfiable(assignment.clone());
        }

        // If any clause is empty, it's unsatisfiable
        if clauses.iter().any(|c| c.is_empty()) {
            return SatResult::Unsatisfiable;
        }

        // Unit propagation: if there's a unit clause, assign it
        if let Some((unit_var, unit_value)) = self.find_unit_clause(clauses) {
            let mut new_assignment = assignment.clone();
            new_assignment.insert(unit_var.clone(), unit_value);
            let simplified = self.simplify_clauses(clauses, &unit_var, unit_value);
            return self.dpll(&simplified, &new_assignment);
        }

        // Pure literal elimination: if a variable appears only positively or only negatively
        if let Some((pure_var, pure_value)) = self.find_pure_literal(clauses) {
            let mut new_assignment = assignment.clone();
            new_assignment.insert(pure_var.clone(), pure_value);
            let simplified = self.simplify_clauses(clauses, &pure_var, pure_value);
            return self.dpll(&simplified, &new_assignment);
        }

        // Choose a variable to branch on
        if let Some(var) = self.choose_variable(clauses) {
            // Try assigning true
            let mut assignment_true = assignment.clone();
            assignment_true.insert(var.clone(), true);
            let simplified_true = self.simplify_clauses(clauses, &var, true);
            let result_true = self.dpll(&simplified_true, &assignment_true);
            if result_true.is_satisfiable() {
                return result_true;
            }

            // Try assigning false
            let mut assignment_false = assignment.clone();
            assignment_false.insert(var.clone(), false);
            let simplified_false = self.simplify_clauses(clauses, &var, false);
            return self.dpll(&simplified_false, &assignment_false);
        }

        // No variables left, but still have clauses - should not happen
        SatResult::Unsatisfiable
    }

    /// Find a unit clause (clause with only one literal)
    fn find_unit_clause(&self, clauses: &[Clause]) -> Option<(Variable, bool)> {
        for clause in clauses {
            if clause.is_unit() {
                let lit = &clause.literals[0];
                return Some((lit.variable.clone(), !lit.negated));
            }
        }
        None
    }

    /// Find a pure literal (variable that appears only positively or only negatively)
    fn find_pure_literal(&self, clauses: &[Clause]) -> Option<(Variable, bool)> {
        let mut positive = HashSet::new();
        let mut negative = HashSet::new();

        for clause in clauses {
            for lit in &clause.literals {
                if lit.negated {
                    negative.insert(lit.variable.clone());
                } else {
                    positive.insert(lit.variable.clone());
                }
            }
        }

        // Find variables that appear only positively
        for var in &positive {
            if !negative.contains(var) {
                return Some((var.clone(), true));
            }
        }

        // Find variables that appear only negatively
        for var in &negative {
            if !positive.contains(var) {
                return Some((var.clone(), false));
            }
        }

        None
    }

    /// Choose a variable to branch on (simple heuristic: first unassigned variable)
    fn choose_variable(&self, clauses: &[Clause]) -> Option<Variable> {
        for clause in clauses {
            for lit in &clause.literals {
                return Some(lit.variable.clone());
            }
        }
        None
    }

    /// Simplify clauses given a variable assignment
    fn simplify_clauses(&self, clauses: &[Clause], var: &Variable, value: bool) -> Vec<Clause> {
        let mut simplified = Vec::new();

        for clause in clauses {
            // Check if this clause is satisfied by the assignment
            let mut satisfied = false;
            let mut new_literals = Vec::new();

            for lit in &clause.literals {
                if &lit.variable == var {
                    // This literal is assigned
                    if lit.negated == !value {
                        // The literal is true, so the clause is satisfied
                        satisfied = true;
                        break;
                    }
                    // Otherwise, the literal is false, so we skip it
                } else {
                    // Keep literals for other variables
                    new_literals.push(lit.clone());
                }
            }

            if !satisfied {
                simplified.push(Clause::new(new_literals));
            }
        }

        simplified
    }
}

/// Solve a CNF formula
pub fn solve_cnf(cnf: &Cnf) -> SatResult {
    let solver = SatSolver::new(cnf.clone());
    solver.solve()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnf::Literal;
    use crate::formula::Formula;

    #[test]
    fn test_satisfiable_simple() {
        // x ∨ y
        let formula = Formula::or(vec![Formula::var("x"), Formula::var("y")]);
        let cnf = Cnf::from_formula(&formula);
        let result = solve_cnf(&cnf);

        assert!(result.is_satisfiable());
    }

    #[test]
    fn test_unsatisfiable() {
        // x ∧ ¬x
        let formula = Formula::and(vec![Formula::var("x"), Formula::not(Formula::var("x"))]);
        let cnf = Cnf::from_formula(&formula);
        let result = solve_cnf(&cnf);

        assert!(result.is_unsatisfiable());
    }

    #[test]
    fn test_satisfiable_complex() {
        // (x ∨ y) ∧ (¬x ∨ z) ∧ (¬y ∨ ¬z)
        let formula = Formula::and(vec![
            Formula::or(vec![Formula::var("x"), Formula::var("y")]),
            Formula::or(vec![Formula::not(Formula::var("x")), Formula::var("z")]),
            Formula::or(vec![
                Formula::not(Formula::var("y")),
                Formula::not(Formula::var("z")),
            ]),
        ]);
        let cnf = Cnf::from_formula(&formula);
        let result = solve_cnf(&cnf);

        assert!(result.is_satisfiable());

        // Verify the assignment satisfies the formula
        if let Some(assignment) = result.assignment() {
            assert_eq!(formula.evaluate(assignment), Some(true));
        }
    }

    #[test]
    fn test_unit_clause_propagation() {
        // x ∧ (x ∨ y)
        let formula = Formula::and(vec![
            Formula::var("x"),
            Formula::or(vec![Formula::var("x"), Formula::var("y")]),
        ]);
        let cnf = Cnf::from_formula(&formula);
        let result = solve_cnf(&cnf);

        assert!(result.is_satisfiable());
        if let Some(assignment) = result.assignment() {
            assert_eq!(assignment.get(&Variable::new("x")), Some(&true));
        }
    }

    #[test]
    fn test_pure_literal_elimination() {
        // x ∧ (x ∨ y)
        // x is a pure literal (appears only positively)
        let formula = Formula::and(vec![
            Formula::var("x"),
            Formula::or(vec![Formula::var("x"), Formula::var("y")]),
        ]);
        let cnf = Cnf::from_formula(&formula);
        let solver = SatSolver::new(cnf.clone());

        let pure = solver.find_pure_literal(&cnf.clauses);
        assert!(pure.is_some());
    }

    #[test]
    fn test_find_unit_clause() {
        let clause1 = Clause::new(vec![Literal::positive(Variable::new("x"))]);
        let clause2 = Clause::new(vec![
            Literal::positive(Variable::new("y")),
            Literal::negative(Variable::new("z")),
        ]);

        let cnf = Cnf::new(vec![clause1, clause2]);
        let solver = SatSolver::new(cnf.clone());

        let unit = solver.find_unit_clause(&cnf.clauses);
        assert!(unit.is_some());
        assert_eq!(unit.unwrap(), (Variable::new("x"), true));
    }

    #[test]
    fn test_simplify_clauses() {
        let clause1 = Clause::new(vec![
            Literal::positive(Variable::new("x")),
            Literal::positive(Variable::new("y")),
        ]);
        let clause2 = Clause::new(vec![
            Literal::negative(Variable::new("x")),
            Literal::positive(Variable::new("z")),
        ]);

        let cnf = Cnf::new(vec![clause1, clause2]);
        let solver = SatSolver::new(cnf.clone());

        // Assign x = true
        let simplified = solver.simplify_clauses(&cnf.clauses, &Variable::new("x"), true);

        // clause1 is satisfied (x is true), so it's removed
        // clause2 becomes just [z]
        assert_eq!(simplified.len(), 1);
        assert_eq!(simplified[0].literals.len(), 1);
    }

    #[test]
    fn test_tautology_detection() {
        // x ∨ ¬x is a tautology
        let formula = Formula::or(vec![Formula::var("x"), Formula::not(Formula::var("x"))]);
        let cnf = Cnf::from_formula(&formula);
        let result = solve_cnf(&cnf);

        assert!(result.is_satisfiable());
    }
}
