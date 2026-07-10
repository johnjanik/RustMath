;; Highlighting queries

;; The highlight group names (e.g. @keyword.function)
;; are specific to neovim, but queries are quite general.

(comment) @comment

";" @delimiter

[
 "."
 ".."
 ","
 ] @punctuation.special

[
  "("
  ")"
  "["
  "]"
  "{"
  "}"
] @punctuation.bracket

(string) @string

(identifier) @variable

(integer) @number
(real) @number.float

(call
 function: (identifier) @function)

;; types 
(type) @type
(typed_identifier
 "::" @operator)



;; constructors

(constructor
 name: (identifier) @variable.builtin
 ["<" ">"] @punctuation.special
 (constructor_elements
  "|" @punctuation.special)?
  (constructor_options
   ":" @punctuation.special)?
 )

;; definitions

(assignment
":=" @operator)

(intrinsic_definition
  "intrinsic" @keyword.function
 docstring: (doc_string) @string.documentation
 ["intrinsic" "end"] @keyword.function
 )

(function_definition
 ["function" "end"] @keyword.function)

(function_definition
 name: (identifier) @variable)

(procedure_definition
 ["procedure" "end"] @keyword.function)

(return_statement
 "return" @keyword.return)

[(break_statement)
 (continue_statement)] @keyword.repeat




;; directives and special statements

[(clear) 
 (freeze)
 (delete)
 (load_directive)
 (save_directive)
 (import_directive)
 ] @keyword.directive

(declare_statement
 "declare" @keyword.directive)

(attribute_declaration
 "attributes" @attribute.declaration)

(type_declaration
 "type" @attribute.declaration)

(verbosity_declaration
 "verbose" @attribute.declaration)


[(print_statement)
 (vprint_statement)
 (read_statement)
 (print_level_statement)
 (assert_statement)
 (require_statement)
 (time_statement)
 (vtime_statement)
 (local_statement)
  ] @keyword.directive


;; expressions

(eval_expression
 "eval" @keyword.return)


;; control flow

(for_statement
 ["for" "do" "end"] @keyword.repeat)

(for_quantifier
 [":=" "to" "by" "in"] @keyword.repeat)

(while_statement
 ["while" "end"] @keyword.repeat)

(repeat_statement
 ["repeat" "end"] @keyword.repeat)

(if_statement
 ["if" "end" "then" "elif" "else"] @keyword.conditional)

(case_statement
 ["case" "end" "else"] @keyword.conditional)

(ternary_operator
 ["select" "else"] @keyword.conditional.ternary)


;; operators

(attribute) @attribute

;; distinguish between symbol operators (+, * ...) and
;; word operators (in, div,...)

(
 (binary_operator
  operator: _ @keyword.operator)
 (#match? @keyword.operator "[a-zA-Z]+")
 )

(
 (binary_operator
  operator: _ @operator)
 (#not-match? @operator "[a-zA-Z]+")
 )

(
 (unary_operator
  operator: _ @keyword.operator)
 (#match? @keyword.operator "[a-zA-Z]+")
 )

(
 (unary_operator
  operator: _ @operator)
 (#not-match? @operator "[a-zA-Z]+")
 )

[
 (true) 
 (false)
 ] @boolean


