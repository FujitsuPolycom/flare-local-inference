-------------------------- MODULE DispatchRing --------------------------
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS PeerCount, MaxEpochs
VARIABLE state

Peers == 0..(PeerCount - 1)
Epochs == 1..MaxEpochs
CoordPhases == {"Idle", "WaitingReady", "WaitingCancel", "Draining", "LaunchingLocal"}
PeerPhases == {
    "WaitingStep", "ReadingPrepare", "WaitingDecision",
    "ReadingCommit", "Draining", "Closed", "Abandoned"
}

Initial == [
    prepareEpoch |-> 0,
    decisionEpoch |-> 0,
    decisionCommit |-> FALSE,
    drain |-> FALSE,
    peerExited |-> FALSE,
    coordPhase |-> "Idle",
    coordEpoch |-> 1,
    pending |-> {},
    committed |-> {},
    peerPhase |-> [p \in Peers |-> "WaitingStep"],
    peerEpoch |-> [p \in Peers |-> 1],
    peerAck |-> [p \in Peers |-> 0],
    peerInFlight |-> [p \in Peers |-> {}],
    slotWriteConflict |-> FALSE,
    strandedOnDrain |-> FALSE,
    abandonedInFlight |-> FALSE,
    localLaunchBeforeCommit |-> FALSE
]

Init == state = Initial

PeerReadsSlot(s, p, slot) ==
    /\ s.peerPhase[p] \in {"ReadingPrepare", "ReadingCommit"}
    /\ (s.peerEpoch[p] % 2) = slot

DecisionEvent(s, e) ==
    IF s.decisionEpoch >= e THEN (IF s.decisionCommit THEN "Commit" ELSE "Cancel")
    ELSE IF s.drain THEN "Drain"
    ELSE "None"

CoordPrepare(before, after) ==
    /\ before.coordPhase = "Idle"
    /\ ~before.drain
    /\ ~before.peerExited
    /\ before.coordEpoch <= MaxEpochs
    /\ Cardinality(before.pending) < 2
    /\ LET slot == before.coordEpoch % 2
           conflict == \E p \in Peers : PeerReadsSlot(before, p, slot)
       IN after = [before EXCEPT
           !.prepareEpoch = before.coordEpoch,
           !.coordPhase = "WaitingReady",
           !.slotWriteConflict = @ \/ conflict]

PeerObserveStep(before, after, p) ==
    /\ before.peerPhase[p] = "WaitingStep"
    /\ ~before.drain
    /\ before.prepareEpoch >= before.peerEpoch[p]
    /\ after = [before EXCEPT !.peerPhase[p] = "ReadingPrepare"]

PeerValidateOk(before, after, p) ==
    /\ before.peerPhase[p] = "ReadingPrepare"
    /\ after = [before EXCEPT
        !.peerPhase[p] = "WaitingDecision",
        !.peerAck[p] = 2 * before.peerEpoch[p]]

PeerValidateFail(before, after, p) ==
    /\ before.peerPhase[p] = "ReadingPrepare"
    /\ after = [before EXCEPT
        !.peerExited = TRUE,
        !.peerPhase[p] = IF before.peerInFlight[p] = {} THEN "Abandoned" ELSE "Draining"]

CoordCommit(before, after) ==
    /\ before.coordPhase = "WaitingReady"
    /\ ~before.peerExited
    /\ \A p \in Peers : before.peerAck[p] >= 2 * before.coordEpoch
    /\ after = [before EXCEPT
        !.decisionEpoch = before.coordEpoch,
        !.decisionCommit = TRUE,
        !.coordPhase = "LaunchingLocal",
        !.committed = @ \cup {before.coordEpoch}]

CoordLaunchLocal(before, after) ==
    /\ before.coordPhase = "LaunchingLocal"
    /\ ~before.drain
    /\ LET premature == (before.decisionEpoch < before.coordEpoch) \/ ~before.decisionCommit
       IN after = [before EXCEPT
           !.coordPhase = "Idle",
           !.coordEpoch = @ + 1,
           !.pending = @ \cup {before.coordEpoch},
           !.localLaunchBeforeCommit = @ \/ premature]

CoordLaunchLocalFail(before, after) ==
    /\ before.coordPhase = "LaunchingLocal"
    /\ ~before.drain
    /\ after = [before EXCEPT
        !.drain = TRUE,
        !.coordPhase = "Draining",
        !.coordEpoch = @ + 1]

CoordCancel(before, after) ==
    /\ before.coordPhase = "WaitingReady"
    /\ ~before.peerExited
    /\ \A p \in Peers : before.peerAck[p] >= 2 * before.coordEpoch
    /\ after = [before EXCEPT
        !.decisionEpoch = before.coordEpoch,
        !.decisionCommit = FALSE,
        !.coordPhase = "WaitingCancel"]

CoordCancelComplete(before, after) ==
    /\ before.coordPhase = "WaitingCancel"
    /\ ~before.peerExited
    /\ \A p \in Peers : before.peerAck[p] >= 2 * before.coordEpoch + 1
    /\ after = [before EXCEPT
        !.coordPhase = "Idle",
        !.coordEpoch = @ + 1]

CoordRequestDrain(before, after) ==
    /\ ~before.drain
    /\ \/ (before.coordPhase = "WaitingReady"
           /\ (before.peerExited \/ (\A p \in Peers : before.peerAck[p] >= 2 * before.coordEpoch)))
       \/ (before.coordPhase = "WaitingCancel" /\ before.peerExited)
       \/ (before.coordPhase = "Idle" /\ (before.pending = {} \/ before.peerExited))
    /\ after = [before EXCEPT
        !.drain = TRUE,
        !.coordPhase = "Draining"]

PeerObserveCommit(before, after, p) ==
    /\ before.peerPhase[p] = "WaitingDecision"
    /\ DecisionEvent(before, before.peerEpoch[p]) = "Commit"
    /\ after = [before EXCEPT !.peerPhase[p] = "ReadingCommit"]

PeerObserveCancel(before, after, p) ==
    /\ before.peerPhase[p] = "WaitingDecision"
    /\ DecisionEvent(before, before.peerEpoch[p]) = "Cancel"
    /\ after = [before EXCEPT
        !.peerAck[p] = 2 * before.peerEpoch[p] + 1,
        !.peerEpoch[p] = @ + 1,
        !.peerPhase[p] = "WaitingStep"]

PeerObserveDecisionDrain(before, after, p) ==
    /\ before.peerPhase[p] = "WaitingDecision"
    /\ DecisionEvent(before, before.peerEpoch[p]) = "Drain"
    /\ after = [before EXCEPT
        !.peerPhase[p] = "Draining",
        !.strandedOnDrain = @ \/ (before.peerEpoch[p] \in before.committed)]

PeerSubmitCommitted(before, after, p) ==
    /\ before.peerPhase[p] = "ReadingCommit"
    /\ after = [before EXCEPT
        !.peerInFlight[p] = @ \cup {before.peerEpoch[p]},
        !.peerEpoch[p] = @ + 1,
        !.peerPhase[p] = "WaitingStep"]

PeerObserveStepDrain(before, after, p) ==
    /\ before.peerPhase[p] = "WaitingStep"
    /\ before.drain
    /\ after = [before EXCEPT
        !.peerPhase[p] = "Draining",
        !.strandedOnDrain = @ \/ (before.peerEpoch[p] \in before.committed)]

PeerCompleteFrame(before, after, p, e) ==
    /\ before.peerPhase[p] \in {"WaitingStep", "ReadingPrepare", "WaitingDecision", "ReadingCommit", "Draining"}
    /\ e \in before.peerInFlight[p]
    /\ \A other \in before.peerInFlight[p] : e <= other
    /\ after = [before EXCEPT !.peerInFlight[p] = @ \ {e}]

PeerFinishDrain(before, after, p) ==
    /\ before.peerPhase[p] = "Draining"
    /\ before.peerInFlight[p] = {}
    /\ after = [before EXCEPT
        !.peerPhase[p] = "Closed",
        !.abandonedInFlight = @ \/ (before.peerInFlight[p] # {})]

CoordRetireFront(before, after, e) ==
    /\ e \in before.pending
    /\ \A other \in before.pending : e <= other
    /\ \A p \in Peers : before.peerEpoch[p] > e /\ e \notin before.peerInFlight[p]
    /\ after = [before EXCEPT !.pending = @ \ {e}]

Next ==
    \/ CoordPrepare(state, state')
    \/ (\E p \in Peers : PeerObserveStep(state, state', p))
    \/ (\E p \in Peers : PeerValidateOk(state, state', p))
    \/ (\E p \in Peers : PeerValidateFail(state, state', p))
    \/ CoordCommit(state, state')
    \/ CoordLaunchLocal(state, state')
    \/ CoordLaunchLocalFail(state, state')
    \/ CoordCancel(state, state')
    \/ CoordCancelComplete(state, state')
    \/ CoordRequestDrain(state, state')
    \/ (\E p \in Peers : PeerObserveCommit(state, state', p))
    \/ (\E p \in Peers : PeerObserveCancel(state, state', p))
    \/ (\E p \in Peers : PeerObserveDecisionDrain(state, state', p))
    \/ (\E p \in Peers : PeerSubmitCommitted(state, state', p))
    \/ (\E p \in Peers : PeerObserveStepDrain(state, state', p))
    \/ (\E p \in Peers, e \in Epochs : PeerCompleteFrame(state, state', p, e))
    \/ (\E p \in Peers : PeerFinishDrain(state, state', p))
    \/ (\E e \in Epochs : CoordRetireFront(state, state', e))
    \/ UNCHANGED state

TypeOK ==
    /\ state.prepareEpoch \in 0..MaxEpochs
    /\ state.decisionEpoch \in 0..MaxEpochs
    /\ state.decisionCommit \in BOOLEAN
    /\ state.drain \in BOOLEAN
    /\ state.peerExited \in BOOLEAN
    /\ state.coordPhase \in CoordPhases
    /\ state.coordEpoch \in 1..(MaxEpochs + 1)
    /\ state.pending \subseteq Epochs
    /\ state.committed \subseteq Epochs
    /\ state.peerPhase \in [Peers -> PeerPhases]
    /\ state.peerEpoch \in [Peers -> 1..(MaxEpochs + 1)]
    /\ state.peerAck \in [Peers -> 0..(2 * MaxEpochs + 1)]
    /\ state.peerInFlight \in [Peers -> SUBSET Epochs]
    /\ state.slotWriteConflict \in BOOLEAN
    /\ state.strandedOnDrain \in BOOLEAN
    /\ state.abandonedInFlight \in BOOLEAN
    /\ state.localLaunchBeforeCommit \in BOOLEAN

CommitRequiresEveryPeer ==
    \A e \in state.committed :
        \A p \in Peers :
            state.peerEpoch[p] > e \/ state.peerAck[p] >= 2 * e

CommitBeforeLocalLaunch == ~state.localLaunchBeforeCommit

SlotReuseSafe == ~state.slotWriteConflict

NoStrandedCollective ==
    /\ ~state.strandedOnDrain
    /\ \A p \in Peers :
        state.peerPhase[p] \in {"Draining", "Closed", "Abandoned"} =>
            (\A e \in state.committed : state.peerEpoch[p] > e)

NoAbandonedInFlightWork ==
    /\ ~state.abandonedInFlight
    /\ \A p \in Peers :
        state.peerPhase[p] \in {"Closed", "Abandoned"} => state.peerInFlight[p] = {}

WindowAndOrder ==
    /\ Cardinality(state.pending) <= 2
    /\ state.pending \subseteq state.committed
    /\ \A p \in Peers :
        /\ state.peerInFlight[p] \subseteq state.committed
        /\ \A e \in state.peerInFlight[p] : e < state.peerEpoch[p]

Spec == Init /\ [][Next]_state

\* Projection encoding for Rust observed-edge conformance (PeerCount = 2, MaxEpochs = 3).
BoolInt(b) == IF b THEN 1 ELSE 0
IntBool(i) == i = 1

DecodeCoordPhase(code) ==
    CASE code = 0 -> "Idle"
      [] code = 1 -> "WaitingReady"
      [] code = 2 -> "WaitingCancel"
      [] code = 3 -> "Draining"
      [] code = 4 -> "LaunchingLocal"

DecodePeerPhase(code) ==
    CASE code = 0 -> "WaitingStep"
      [] code = 1 -> "ReadingPrepare"
      [] code = 2 -> "WaitingDecision"
      [] code = 3 -> "ReadingCommit"
      [] code = 4 -> "Draining"
      [] code = 5 -> "Closed"
      [] code = 6 -> "Abandoned"

DecodeMask(mask) == {e \in Epochs : (mask \div (2 ^ (e - 1))) % 2 = 1}

Decode(vec) == [
    prepareEpoch |-> vec[1],
    decisionEpoch |-> vec[2],
    decisionCommit |-> IntBool(vec[3]),
    drain |-> IntBool(vec[4]),
    peerExited |-> IntBool(vec[5]),
    coordPhase |-> DecodeCoordPhase(vec[6]),
    coordEpoch |-> vec[7],
    pending |-> DecodeMask(vec[8]),
    committed |-> DecodeMask(vec[9]),
    peerPhase |-> [p \in Peers |-> IF p = 0 THEN DecodePeerPhase(vec[10]) ELSE DecodePeerPhase(vec[14])],
    peerEpoch |-> [p \in Peers |-> IF p = 0 THEN vec[11] ELSE vec[15]],
    peerAck |-> [p \in Peers |-> IF p = 0 THEN vec[12] ELSE vec[16]],
    peerInFlight |-> [p \in Peers |-> IF p = 0 THEN DecodeMask(vec[13]) ELSE DecodeMask(vec[17])],
    slotWriteConflict |-> (vec[18] % 2) = 1,
    strandedOnDrain |-> ((vec[18] \div 2) % 2) = 1,
    abandonedInFlight |-> ((vec[18] \div 4) % 2) = 1,
    localLaunchBeforeCommit |-> ((vec[18] \div 8) % 2) = 1
]

AllowedEdge(edge) ==
    LET before == Decode(edge[1])
        action == edge[2]
        after == Decode(edge[3])
    IN CASE action[1] = 1 -> CoordPrepare(before, after)
         [] action[1] = 2 -> PeerObserveStep(before, after, action[2])
         [] action[1] = 3 -> PeerValidateOk(before, after, action[2])
         [] action[1] = 4 -> PeerValidateFail(before, after, action[2])
         [] action[1] = 5 -> CoordCommit(before, after)
         [] action[1] = 6 -> CoordCancel(before, after)
         [] action[1] = 7 -> CoordCancelComplete(before, after)
         [] action[1] = 8 -> CoordRequestDrain(before, after)
         [] action[1] = 9 -> PeerObserveCommit(before, after, action[2])
         [] action[1] = 10 -> PeerObserveCancel(before, after, action[2])
         [] action[1] = 11 -> PeerObserveDecisionDrain(before, after, action[2])
         [] action[1] = 12 -> PeerSubmitCommitted(before, after, action[2])
         [] action[1] = 13 -> PeerObserveStepDrain(before, after, action[2])
         [] action[1] = 14 -> PeerCompleteFrame(before, after, action[2], action[3])
         [] action[1] = 15 -> PeerFinishDrain(before, after, action[2])
         [] action[1] = 16 -> CoordRetireFront(before, after, action[2])
         [] action[1] = 17 -> CoordLaunchLocal(before, after)
         [] action[1] = 18 -> CoordLaunchLocalFail(before, after)
         [] OTHER -> FALSE
=============================================================================
