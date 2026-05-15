# Development Notes

## Node State Machine

DIRTY -> PENDING -> COMPUTING -> CLEAN
  ^                                |
  |________________________________|
         (on dependency change)

PENDING and COMPUTING states are reserved for future async execution.
Currently only CLEAN and DIRTY are used in the synchronous engine.
