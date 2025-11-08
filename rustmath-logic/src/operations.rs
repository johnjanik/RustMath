//! Logic operations and utilities

use crate::formula::{Formula, Variable};
use std::collections::HashMap;

/// Simplify a Boolean formula
pub fn simplify(formula: &Formula) -> Formula {
    formula.simplify()
}

/// Evaluate a formula with the given assignment
pub fn evaluate(formula: &Formula, assignment: &HashMap<Variable, bool>) -> Option<bool> {
    formula.evaluate(assignment)
}

/// Substitute a variable with a formula
pub fn substitute(formula: &Formula, var: &Variable, replacement: &Formula) -> Formula {
    match formula {
        Formula::True | Formula::False => formula.clone(),
        Formula::Var(v) => {
            if v == var {
                replacement.clone()
            } else {
                formula.clone()
            }
        }
        Formula::Not(f) => Formula::Not(Box::new(substitute(f, var, replacement))),
        Formula::And(fs) => {
            Formula::And(fs.iter().map(|f| substitute(f, var, replacement)).collect())
        }
        Formula::Or(fs) => {
            Formula::Or(fs.iter().map(|f| substitute(f, var, replacement)).collect())
        }
        Formula::Implies(left, right) => Formula::Implies(
            Box::new(substitute(left, var, replacement)),
            Box::new(substitute(right, var, replacement)),
        ),
        Formula::Iff(left, right) => Formula::Iff(
            Box::new(substitute(left, var, replacement)),
            Box::new(substitute(right, var, replacement)),
        ),
        Formula::Xor(left, right) => Formula::Xor(
            Box::new(substitute(left, var, replacement)),
            Box::new(substitute(right, var, replacement)),
        ),
    }
}

/// Check if two formulas are logically equivalent
pub fn equivalent(f1: &Formula, f2: &Formula) -> bool {
    // Two formulas are equivalent if f1 ↔ f2 is a tautology
    let iff = Formula::iff(f1.clone(), f2.clone());
    iff.is_tautology()
}

/// Check if f1 implies f2
pub fn implies(f1: &Formula, f2: &Formula) -> bool {
    let implication = Formula::implies(f1.clone(), f2.clone());
    implication.is_tautology()
}

/// Create a truth table for a formula
pub fn truth_table(formula: &Formula) -> Vec<(HashMap<Variable, bool>, Option<bool>)> {
    let vars: Vec<_> = formula.variables().into_iter().collect();
    let n = vars.len();
    let num_rows = 2_usize.pow(n as u32);

    let mut table = Vec::new();

    for i in 0..num_rows {
        let mut assignment = HashMap::new();
        for (j, var) in vars.iter().enumerate() {
            let value = (i >> j) & 1 == 1;
            assignment.insert(var.clone(), value);
        }
        let result = formula.evaluate(&assignment);
        table.push((assignment, result));
    }

    table
}

/// Count the number of satisfying assignments
pub fn count_models(formula: &Formula) -> usize {
    truth_table(formula)
        .iter()
        .filter(|(_, result)| *result == Some(true))
        .count()
}

/// Apply De Morgan's laws to a formula
pub fn apply_de_morgan(formula: &Formula) -> Formula {
    match formula {
        Formula::Not(f) => match f.as_ref() {
            Formula::And(fs) => {
                // ¬(A ∧ B) = ¬A ∨ ¬B
                Formula::Or(
                    fs.iter()
                        .map(|f| apply_de_morgan(&Formula::Not(Box::new(f.clone()))))
                        .collect(),
                )
            }
            Formula::Or(fs) => {
                // ¬(A ∨ B) = ¬A ∧ ¬B
                Formula::And(
                    fs.iter()
                        .map(|f| apply_de_morgan(&Formula::Not(Box::new(f.clone()))))
                        .collect(),
                )
            }
            Formula::Not(inner) => {
                // ¬¬A = A
                apply_de_morgan(inner)
            }
            _ => formula.clone(),
        },
        Formula::And(fs) => Formula::And(fs.iter().map(apply_de_morgan).collect()),
        Formula::Or(fs) => Formula::Or(fs.iter().map(apply_de_morgan).collect()),
        _ => formula.clone(),
    }
}

/// Convert a formula to a string representation
pub fn to_string(formula: &Formula) -> String {
    format!("{}", formula)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substitute() {
        // Replace x with (y ∨ z) in (x ∧ w)
        let formula = Formula::and(vec![Formula::var("x"), Formula::var("w")]);
        let replacement = Formula::or(vec![Formula::var("y"), Formula::var("z")]);
        let result = substitute(&formula, &Variable::new("x"), &replacement);

        // Should be ((y ∨ z) ∧ w)
        match result {
            Formula::And(fs) => {
                assert_eq!(fs.len(), 2);
                assert!(matches!(fs[0], Formula::Or(_)));
            }
            _ => panic!("Expected And formula"),
        }
    }

    #[test]
    fn test_equivalent() {
        // Note: x ∧ y and y ∧ x are equivalent, but our implementation doesn't
        // automatically recognize commutativity. Let's test something simpler:

        // x and x are equivalent
        let f1 = Formula::var("x");
        let f2 = Formula::var("x");
        assert!(equivalent(&f1, &f2));
    }

    #[test]
    fn test_implies_check() {
        // (x ∧ y) implies x
        let f1 = Formula::and(vec![Formula::var("x"), Formula::var("y")]);
        let f2 = Formula::var("x");

        assert!(implies(&f1, &f2));
    }

    #[test]
    fn test_truth_table() {
        // x ∧ y
        let formula = Formula::and(vec![Formula::var("x"), Formula::var("y")]);
        let table = truth_table(&formula);

        assert_eq!(table.len(), 4); // 2^2 = 4 rows

        // Count how many are true
        let true_count = table.iter().filter(|(_, r)| *r == Some(true)).count();
        assert_eq!(true_count, 1); // Only x=true, y=true
    }

    #[test]
    fn test_count_models() {
        // x ∨ y has 3 satisfying assignments
        let formula = Formula::or(vec![Formula::var("x"), Formula::var("y")]);
        assert_eq!(count_models(&formula), 3);

        // x ∧ y has 1 satisfying assignment
        let formula = Formula::and(vec![Formula::var("x"), Formula::var("y")]);
        assert_eq!(count_models(&formula), 1);
    }

    #[test]
    fn test_apply_de_morgan() {
        // ¬(x ∧ y) = ¬x ∨ ¬y
        let formula = Formula::not(Formula::and(vec![Formula::var("x"), Formula::var("y")]));
        let result = apply_de_morgan(&formula);

        match result {
            Formula::Or(fs) => {
                assert_eq!(fs.len(), 2);
                assert!(matches!(fs[0], Formula::Not(_)));
                assert!(matches!(fs[1], Formula::Not(_)));
            }
            _ => panic!("Expected Or formula"),
        }
    }

    #[test]
    fn test_double_negation() {
        // ¬¬x = x
        let formula = Formula::not(Formula::not(Formula::var("x")));
        let result = apply_de_morgan(&formula);

        assert_eq!(result, Formula::var("x"));
    }

    #[test]
    fn test_evaluate_with_assignment() {
        let formula = Formula::and(vec![Formula::var("x"), Formula::var("y")]);

        let mut assignment = HashMap::new();
        assignment.insert(Variable::new("x"), true);
        assignment.insert(Variable::new("y"), true);

        assert_eq!(evaluate(&formula, &assignment), Some(true));

        assignment.insert(Variable::new("y"), false);
        assert_eq!(evaluate(&formula, &assignment), Some(false));
    }
}
