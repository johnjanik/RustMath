;; Local variable scoping queries for Magma

;; Scope boundaries
(function_definition) @local.scope
(procedure_definition) @local.scope
(intrinsic_definition) @local.scope
(for_statement) @local.scope
(while_statement) @local.scope
(repeat_statement) @local.scope
(where_expression) @local.scope

;; Definitions — function/procedure parameters
(parameters
  (identifier) @local.definition)

(typed_identifier
  (identifier) @local.definition)

(ref_identifier
  (identifier) @local.definition)

;; Definitions — local statement
(local_statement
  (identifier) @local.definition)

;; Definitions — assignment LHS (only bare identifier assignments, not indexed/sliced)
(assignment
  left: (identifier) @local.definition)

;; Definitions — for loop variables
;; Form 1: for i := expr to expr (identifier is direct child of for_quantifier)
(for_quantifier
  (identifier) @local.definition)

;; Form 2: for i in S (parsed as binary_operator with `in`; left is the loop var)
(for_quantifier
  (binary_operator
    left: (identifier) @local.definition))

;; Definitions — where clause bindings
(where_expression
  variables: (identifier) @local.definition)

;; References
(identifier) @local.reference
