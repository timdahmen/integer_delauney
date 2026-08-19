## File organization

- **No sprawling files.** A file mixing several concerns should be split by concern.
- **Public API leads.** In a file whose job is to expose an API, the API surface 
  goes first and reads as the documentation of what's offered. Implementation detail 
  and helpers move to their own file, or at least to the bottom.
- **One concept, one file.** A type with its own identity (a wrapper class,
  a data structure) gets its own file even if it's currently small.

## Comments
- State what the code **does or returns**, present tense, one line where
  possible. Not what it *used to do*, not why an alternative wasn't chosen,
  not a defense of the design against hypothetical objections — that
  reasoning belongs in the commit message, where it has the diff for
  context, not in the code.
- Doc comments for return values: `returns <shape/format>`.
- A comment that explains an invariant, a non-obvious constraint, or the
  answer to "why is this here and not somewhere more obvious" earns its
  keep. A comment that just restates the next line in words doesn't.

## API documentation
- Every publicly exposed symbol gets at least a one-line doc comment.
- When two exposed names are near-synonyms (`insert` / `insert_deferred`,
  a plain and a `_timed`/`_debug` variant), the doc comment must say when to
  reach for one over the other.

## Duplication
- Important rule Avoid code duplication on all levels. Examples are:
  - functions with similiar names (compute and compute_timed or such)
    suggesting one might wrap the other
  - replicated code  that exists in a superclass or somewhere else, 
    specifically in non-obvious cases where exposing that duplication
	requires some refactoring
  - low level code duplication such as identical conditions in if statements
  - code similiarity that could be exploitet by factoring out new superclasses
    (or more consistently using / extending existing ones)

## No Magic Numbers
- Avoid magic number constants in code: no -1 for DEFAULT, max_int as 
  guardian or such. Use  static constexpr int or enums (C++) respectively
  placeholder variables (Python) with speaking names.
  
## GOLDEN RULE
- If nothing else applies, loosely use the ideas expressed 
  in Clean Code by R. C. Martin as guidelines.
