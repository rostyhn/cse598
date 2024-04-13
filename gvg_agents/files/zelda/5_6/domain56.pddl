(define (domain zelda)
(:requirements :strips :typing :equality)
(:types)
(:predicates (leftOf-cell_0_0-cell_1_0)
             (leftOf-cell_0_1-cell_1_1)
             (leftOf-cell_0_2-cell_1_2)
             (leftOf-cell_0_3-cell_1_3)
             (leftOf-cell_0_4-cell_1_4)
             (leftOf-cell_1_0-cell_2_0)
             (leftOf-cell_1_1-cell_2_1)
             (leftOf-cell_1_2-cell_2_2)
             (leftOf-cell_1_3-cell_2_3)
             (leftOf-cell_1_4-cell_2_4)
             (leftOf-cell_2_0-cell_3_0)
             (leftOf-cell_2_1-cell_3_1)
             (leftOf-cell_2_2-cell_3_2)
             (leftOf-cell_2_3-cell_3_3)
             (leftOf-cell_2_4-cell_3_4)
             (leftOf-cell_3_0-cell_4_0)
             (leftOf-cell_3_1-cell_4_1)
             (leftOf-cell_3_2-cell_4_2)
             (leftOf-cell_3_3-cell_4_3)
             (leftOf-cell_3_4-cell_4_4)
             (leftOf-cell_4_0-cell_5_0)
             (leftOf-cell_4_1-cell_5_1)
             (leftOf-cell_4_2-cell_5_2)
             (leftOf-cell_4_3-cell_5_3)
             (leftOf-cell_4_4-cell_5_4)
             (rightOf-cell_1_0-cell_0_0)
             (rightOf-cell_1_1-cell_0_1)
             (rightOf-cell_1_2-cell_0_2)
             (rightOf-cell_1_3-cell_0_3)
             (rightOf-cell_1_4-cell_0_4)
             (rightOf-cell_2_0-cell_1_0)
             (rightOf-cell_2_1-cell_1_1)
             (rightOf-cell_2_2-cell_1_2)
             (rightOf-cell_2_3-cell_1_3)
             (rightOf-cell_2_4-cell_1_4)
             (rightOf-cell_3_0-cell_2_0)
             (rightOf-cell_3_1-cell_2_1)
             (rightOf-cell_3_2-cell_2_2)
             (rightOf-cell_3_3-cell_2_3)
             (rightOf-cell_3_4-cell_2_4)
             (rightOf-cell_4_0-cell_3_0)
             (rightOf-cell_4_1-cell_3_1)
             (rightOf-cell_4_2-cell_3_2)
             (rightOf-cell_4_3-cell_3_3)
             (rightOf-cell_4_4-cell_3_4)
             (rightOf-cell_5_0-cell_4_0)
             (rightOf-cell_5_1-cell_4_1)
             (rightOf-cell_5_2-cell_4_2)
             (rightOf-cell_5_3-cell_4_3)
             (rightOf-cell_5_4-cell_4_4)
             (above-cell_0_0-cell_0_1)
             (above-cell_0_1-cell_0_2)
             (above-cell_0_2-cell_0_3)
             (above-cell_0_3-cell_0_4)
             (above-cell_1_0-cell_1_1)
             (above-cell_1_1-cell_1_2)
             (above-cell_1_2-cell_1_3)
             (above-cell_1_3-cell_1_4)
             (above-cell_2_0-cell_2_1)
             (above-cell_2_1-cell_2_2)
             (above-cell_2_2-cell_2_3)
             (above-cell_2_3-cell_2_4)
             (above-cell_3_0-cell_3_1)
             (above-cell_3_1-cell_3_2)
             (above-cell_3_2-cell_3_3)
             (above-cell_3_3-cell_3_4)
             (above-cell_4_0-cell_4_1)
             (above-cell_4_1-cell_4_2)
             (above-cell_4_2-cell_4_3)
             (above-cell_4_3-cell_4_4)
             (above-cell_5_0-cell_5_1)
             (above-cell_5_1-cell_5_2)
             (above-cell_5_2-cell_5_3)
             (above-cell_5_3-cell_5_4)
             (below-cell_0_1-cell_0_0)
             (below-cell_0_2-cell_0_1)
             (below-cell_0_3-cell_0_2)
             (below-cell_0_4-cell_0_3)
             (below-cell_1_1-cell_1_0)
             (below-cell_1_2-cell_1_1)
             (below-cell_1_3-cell_1_2)
             (below-cell_1_4-cell_1_3)
             (below-cell_2_1-cell_2_0)
             (below-cell_2_2-cell_2_1)
             (below-cell_2_3-cell_2_2)
             (below-cell_2_4-cell_2_3)
             (below-cell_3_1-cell_3_0)
             (below-cell_3_2-cell_3_1)
             (below-cell_3_3-cell_3_2)
             (below-cell_3_4-cell_3_3)
             (below-cell_4_1-cell_4_0)
             (below-cell_4_2-cell_4_1)
             (below-cell_4_3-cell_4_2)
             (below-cell_4_4-cell_4_3)
             (below-cell_5_1-cell_5_0)
             (below-cell_5_2-cell_5_1)
             (below-cell_5_3-cell_5_2)
             (below-cell_5_4-cell_5_3)
             (at_0-player0-cell_3_1)
             (clear-cell_3_0)
             (clear-cell_3_3)
             (wall-cell_0_3)
             (clear-cell_0_1)
             (has_key-)
             (wall-cell_4_1)
             (clear-cell_5_4)
             (clear-cell_3_4)
             (at_0-player0-cell_3_0)
             (wall-cell_2_3)
             (at_0-player0-cell_0_0)
             (clear-cell_1_2)
             (at_0-player0-cell_2_0)
             (at_1-key0-cell_4_0)
             (next_to_monster-)
             (at_3-door0-cell_1_1)
             (clear-cell_0_0)
             (wall-cell_0_4)
             (clear-cell_2_0)
             (clear-cell_5_3)
             (clear-cell_3_1)
             (monster_alive-monster_0_1)
             (wall-cell_0_2)
             (at_0-player0-cell_4_0)
             (clear-cell_5_2)
             (clear-cell_4_3)
             (wall-cell_4_4)
             (wall-cell_3_2)
             (wall-cell_2_1)
             (escaped-)
             (clear-cell_4_0)
             (at_0-player0-cell_1_0)
             (wall-cell_5_0)
             (clear-cell_1_0)
             (wall-cell_4_2)
             (clear-cell_1_4)
             (clear-cell_5_1)
             (at_0-player0-cell_1_1)
             (clear-cell_1_3)
             (clear-cell_1_1)
             (clear-cell_2_2)
             (clear-cell_2_4)
             (at_2-monster_0_1-cell_0_1))

(:action a0
  :parameters (              )
  :precondition (and (at_0-player0-cell_3_1)
        (clear-cell_3_0)
       )
         :effect (and (not (at_0-player0-cell_3_1)
       )
        (not (clear-cell_3_0)
       )
        (at_0-player0-cell_3_0)
        (clear-cell_3_1)
       ))


       (:action a1
  :parameters (              )
  :precondition (and (at_0-player0-cell_3_0)
        (at_1-key0-cell_4_0)
       )
         :effect (and (not (at_0-player0-cell_3_0)
       )
        (not (at_1-key0-cell_4_0)
       )
        (has_key-)
        (at_0-player0-cell_4_0)
        (clear-cell_3_0)
       ))


       (:action a2
  :parameters (              )
  :precondition (and (at_0-player0-cell_4_0)
        (clear-cell_3_0)
       )
         :effect (and (not (at_0-player0-cell_4_0)
       )
        (not (clear-cell_3_0)
       )
        (at_0-player0-cell_3_0)
        (clear-cell_4_0)
       ))


       (:action a3
  :parameters (              )
  :precondition (and (at_0-player0-cell_3_0)
        (clear-cell_2_0)
       )
         :effect (and (not (at_0-player0-cell_3_0)
       )
        (not (clear-cell_2_0)
       )
        (at_0-player0-cell_2_0)
        (clear-cell_3_0)
       ))


       (:action a4
  :parameters (              )
  :precondition (and (at_0-player0-cell_2_0)
        (clear-cell_1_0)
       )
         :effect (and (not (at_0-player0-cell_2_0)
       )
        (not (clear-cell_1_0)
       )
        (clear-cell_2_0)
        (at_0-player0-cell_1_0)
       ))


       (:action a5
  :parameters (              )
  :precondition (and (at_0-player0-cell_1_0)
        (at_2-monster_0_1-cell_0_1)
        (monster_alive-monster_0_1)
        (clear-cell_0_0)
       )
         :effect (and (not (at_0-player0-cell_1_0)
       )
        (not (clear-cell_0_0)
       )
        (at_0-player0-cell_0_0)
        (next_to_monster-)
        (clear-cell_1_0)
       ))


       (:action a6
  :parameters (              )
  :precondition (and (at_0-player0-cell_0_0)
        (at_2-monster_0_1-cell_0_1)
        (monster_alive-monster_0_1)
        (next_to_monster-)
       )
         :effect (and (not (at_2-monster_0_1-cell_0_1)
       )
        (not (monster_alive-monster_0_1)
       )
        (not (next_to_monster-)
       )
        (clear-cell_0_1)
       ))


       (:action a7
  :parameters (              )
  :precondition (and (at_0-player0-cell_0_0)
        (clear-cell_1_0)
        (not (monster_alive-monster_0_1)
       )
        (not (at_2-monster_0_1-cell_0_1)
       )
       )
         :effect (and (not (at_0-player0-cell_0_0)
       )
        (not (clear-cell_1_0)
       )
        (clear-cell_0_0)
        (at_0-player0-cell_1_0)
       ))


       (:action a8
  :parameters (              )
  :precondition (and (at_0-player0-cell_1_0)
        (at_3-door0-cell_1_1)
        (clear-cell_1_1)
        (has_key-)
        (not (at_1-key0-cell_4_0)
       )
        (not (monster_alive-monster_0_1)
       )
        (not (at_2-monster_0_1-cell_0_1)
       )
       )
         :effect (and (not (at_0-player0-cell_1_0)
       )
        (not (clear-cell_1_1)
       )
        (escaped-)
        (clear-cell_1_0)
        (at_0-player0-cell_1_1)
       ))


       )

       