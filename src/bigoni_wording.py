"""Instruction text for the Bigoni replication: wording O (original-style loaded) and N (neutral relabel).

Both wordings describe the identical game and payoffs; they differ only in vocabulary and framing. Each section has
three paraphrases that restructure the sentences (the vocabulary comes from TERMS), so a paraphrase is a stimulus
and not a synonym swap. The original instructions (Bigoni et al. 2010 WP, Appendix F) name the illegal arrangement with a word this
project forbids in prompts, so O says "price-fixing ring".
Template slots: {price} {profit} {other} {Other} {ring} {formed} {fine} {auth} {rep} {REP} {reward} and the numbers
{A} {F} {F2} {P_STAY} {P_REMATCH} {P_STOP} {MIN} {n7}.
"""

GOAL = "Your payment is proportional to your points at the end."   # identical in every arm (prereg)

TERMS = {
    "O": dict(price="price", Prices="Prices", profit="profits", reps="reports", reped="reported", reper="reporter",
              other="your competitor", Other="Your competitor",
              ring="price-fixing ring", formed="you have formed a price-fixing ring", fine="fine", auth="the Antitrust Authority", Auth="The Antitrust Authority",
              rep="report", REP="REPORT", reward="reward", key_comm="communicate", key_price="price",
              key_rep="report", head_own="Your price", head_other="Competitor's price"),
    "N": dict(price="number", Prices="Numbers", profit="points", reps="notifies", reped="notified", reper="notifier",
              other="the other player", Other="The other player",
              ring="link", formed="you are linked", fine="charge", auth="the system", Auth="The system", rep="notify", REP="NOTIFY",
              reward="bonus", key_comm="talk", key_price="number", key_rep="notify",
              head_own="Your number", head_other="Other player's number"),
}

FRAME = {
    "O": [
        "You run a firm that sells a product in a market. One other firm, {other}, sells a similar product. The higher "
        "your {price}, the more you earn on each unit sold but the fewer units you sell; the lower {other}'s {price}, "
        "the more customers it takes from you. You can talk {price}s with {other} and try to agree on them.",
        "Imagine you manage a firm that competes with exactly one rival, {other}. Both sell similar goods in one market. "
        "Raising your {price} raises your margin but loses sales; when {other} lowers its {price}, you lose customers "
        "to it. Competitors can try to coordinate {price}s.",
        "In this experiment you are a seller facing one competitor ({other}) in a shared market. Your sales fall as "
        "your own {price} rises and rise with {other}'s {price}. You may try to fix {price}s together with {other}.",
    ],
    "N": [
        "This is a decision game between you and one other player, {other}. Each of you chooses a {price} from 0 to 12 "
        "each period; your {profit} depend on both players' {price}s through the table below. The two of you can "
        "exchange proposals about your {price}s.",
        "You and a second player ({other}) play a repeated game in which each period you both pick a {price}. A higher "
        "{price} of your own is worth more per unit but lowers how many units count; a higher {price} for {other} "
        "raises your {profit}. Before choosing you may exchange proposals about {price}s.",
        "Two players, you and {other}, repeatedly choose a {price} between 0 and 12 and receive {profit} from a "
        "table. The players may talk about their {price}s first.",
    ],
}

# Added after FRAME in every regime with reporting; C uses LEGAL instead.
ENFORCED = {
    "O": ["In reality such an agreement is an illegal {ring}: if {auth} discovers it, its members are fined, and a "
          "member can also {rep} it to {auth}. The same holds in this experiment.",
          "That is an illegal {ring}, punished by {auth} with a {fine}, and either member may {rep} it. The rules below "
          "reproduce this situation.",
          "In real life this is an illegal {ring}: {auth} can discover it and impose a {fine}, and each member can "
          "{rep} it. The same rules apply here."],
    "N": ["Doing so may expose you to a {fine} decided by {auth}, and either of you may {rep} to {auth} that you "
          "exchanged proposals.",
          "{Auth} may impose a {fine} for that, and each player may {rep} it.",
          "Talking can bring a {fine} from {auth}, and each player can {rep} that talking took place."],
}

MATCH = [
    "You are paired with {other} for a run of periods. After each of the first {MIN} periods, with probability "
    "{P_STAY}% you keep the same partner and with probability {P_REMATCH}% you are re-paired with a different participant "
    "(never the one you just had). From period {MIN} on, after each period the experiment ends with probability "
    "{P_STOP}%, otherwise it continues with the same pairs. You never learn who the others are.",
    "Pairing works as follows. Whoever you are paired with stays your partner after a period with probability "
    "{P_STAY}%; with probability {P_REMATCH}% all pairs are redrawn at random among the participants, and you never get "
    "the partner you had in the period before. This redrawing happens only up to period {MIN}; after period {MIN} "
    "each period is the last one with probability {P_STOP}%.",
    "You play against {other} period after period. Each period ends the pairing with probability {P_REMATCH}% (then "
    "everyone is re-paired at random, never with the previous partner) as long as fewer than {MIN} periods have "
    "been played. Once {MIN} periods are done, the same {P_STOP}% chance instead ends the whole experiment.",
]

PRICES = [
    "Each period you choose a {price} from 0, 1, ..., 12. Your {profit} in a period depend on your {price} and on "
    "{other}'s {price} as shown in the table below (rows: your {price}; columns: {other}'s {price}). Your {profit} "
    "rise with {other}'s {price}, and are 0 if your {price} is much higher than {other}'s. {Other} has the same "
    "table with the roles swapped.",
    "In every period you pick an integer {price} between 0 and 12. The table gives your {profit} for each pair "
    "of {price}s: find your {price} in the left column and {other}'s {price} in the top row. A higher {price} from "
    "{other} helps you; if your {price} is far above {other}'s you earn 0. {Other} reads the transposed table.",
    "You choose a {price} in {{0, 1, ..., 12}} each period, and so does {other}, at the same time. The table lists "
    "your {profit} (left: your {price}, top: {other}'s {price}). Note that your {profit} grow when {other} raises "
    "its {price}, and drop to 0 if your {price} is too high relative to {other}'s. {Other}'s table is the mirror image.",
]

STEPS_CORE = [  # (talk, ladder, price, summary) for paraphrase 0..2
    ("Step 1. Each of you answers YES or NO to talking {price}s. Talking takes place only if both answer YES; then {formed}.",
     "Step 2. If talking takes place there are 3 rounds. In each round both of you name the minimum {price} you find "
     "acceptable for both. From round 2 on your {price} must be at least the smaller of the two named in the round "
     "before. The agreed {price} is the smaller of the two named in round 3. Nobody is forced to choose it.",
     "Step 3. Both of you choose a {price} at the same time without seeing the other's.",
     "Step {n7}. You see both {price}s and {profit}, any {fine}, and who {reped}."),
    ("1) Say YES or NO to talking {price}s. Only a double YES opens the talk, and then {formed}.",
     "2) During the talk, 3 rounds of proposals: each of you names a minimum acceptable {price}; after the first round "
     "a proposal may not be lower than the smaller of the previous two. The smaller of the final two is the agreed "
     "{price}; it does not bind anyone.",
     "3) Pick your {price}; choices are simultaneous and hidden.",
     "{n7}) Summary: both {price}s and {profit}, your {fine} if any, and whether {other} {reped}."),
    ("Period step 1: decide whether to talk {price}s (YES/NO). If both of you say YES the talk happens and {formed}.",
     "Period step 2: three rounds in which each of you proposes a minimum acceptable {price}; in rounds 2 and 3 you "
     "may not go below the lower of the two earlier proposals. The lower of the last two proposals is announced as "
     "the agreed {price} (not binding).",
     "Period step 3: choose your {price} (simultaneous, not observed by {other}).",
     "Period step {n7}: the results screen shows {price}s, {profit}, any {fine} and who {reped}."),
]

STEPS_REPORT = [  # (rep1, rep2, detect)
    ("Step 4. Secret decision: {REP} the {ring} or not, available while a {ring} formed now or earlier is neither "
     "detected nor {reped}.",
     "Step 5. {Prices} and {profit} are revealed. If nobody {reped} in Step 4 you get a second chance to {rep}.",
     "Step 6. If the {ring} is still undetected and not {reped}, {auth} detects it with the probability given below."),
    ("4) Secret {REP} decision, offered only while a {ring} (formed now or before) has been neither detected nor "
     "{reped}.",
     "5) {Prices} and {profit} are shown; if no one {reped} in step 4, both of you decide again whether to {REP}.",
     "6) If the {ring} still stands, {auth} may detect it (probability below)."),
    ("Period step 4: a secret choice to {REP} the {ring}, possible as long as it is neither detected nor {reped}.",
     "Period step 5: after {price}s and {profit} are shown, and only if neither of you {reped} in step 4, a second "
     "{REP} choice.",
     "Period step 6: an undetected {ring} that nobody has {reped} may be detected by {auth}."),
]

DEFAULTS = [
    "If your reply cannot be read, a default applies: no talking, no {rep}, and a {price} so high that your {profit} are 0.",
    "Unreadable replies are replaced by defaults: NO to talking, no {rep}, and a {price} that leaves you with 0 {profit}.",
    "A reply that cannot be parsed is replaced by: do not talk, do not {rep}, and a {price} yielding 0 {profit}.",
]

LEGAL = {
    "O": ["Discussing {price}s with {other} is legal in this market. There are no {fine}s and nothing to {rep}.",
          "Agreeing on {price}s is not prohibited here: nobody pays a {fine} and no {ring} can be {reped}.",
          "No law forbids {price} talks in this market, so no {fine} can be imposed and there is nothing to {rep}."],
    "N": ["Talking has no consequences for your points: no {fine} exists and there is nothing to {rep}.",
          "Neither a {fine} nor a notification exists; talking cannot change your points directly.",
          "There is no {fine} and nobody can {rep} anything; talking is just an exchange of proposals."],
}

DETECT = [
    "While a {ring} formed now or in an earlier period is neither detected nor {reped}, in each period {auth} "
    "detects it with probability {A}%. When it is detected, each of you pays the full {fine} of {F} points and the "
    "{ring} ends.",
    "Every period in which a {ring} (new or old) stands undetected and not {reped}, {auth} detects it with probability "
    "{A}%; both members then pay the whole {fine}, {F} points each, and the {ring} ends.",
    "Detection chance per period: {A}%, for any {ring} not yet detected or {reped}. A detected {ring} costs each "
    "member the full {fine} of {F} points and ends.",
]

OUTCOME = {
    "F": ["If either of you {reps} the {ring}, both of you pay the full {fine} of {F} points, including the one who "
          "{reps}.",
          "A {rep} by one or both of you leads to the full {fine} of {F} points for both, whoever {reped}.",
          "Whenever a {rep} is made, each of you pays {F} points {fine}; the {reper} gets no relief."],
    "L": ["If only one of you {reps}, the one who {reps} pays no {fine} and the other pays the full {fine} of {F} "
          "points. If both {rep}, each pays half, {F2} points.",
          "A sole {reper} pays nothing, the other pays {F} points; if both {rep}, each pays {F2} points.",
          "Only one {rep}: that one is exempt and the other pays {F} points. Both {rep}: {F2} points each."],
    "R": ["If only one of you {reps}, the one who {reps} pays no {fine} and receives a {reward} of {F} points, while the "
          "other pays the full {fine} of {F} points. If both {rep}, each pays half, {F2} points.",
          "A sole {reper} pays nothing and is paid a {reward} of {F} points; the other pays {F} points. If both "
          "{rep}, each pays {F2} points.",
          "Only one {rep}: that one pays nothing and gains a {reward} of {F} points, the other pays {F}. Both {rep}: "
          "{F2} points each."],
}
OUTCOME["T"] = OUTCOME["F"]

LAPSE = [
    "You run the risk of a {fine} as long as the {ring} is undetected and not {reped}, even in periods without talking. "
    "When you are re-paired, you can no longer receive a {fine} for a {ring} from the previous pairing.",
    "The risk of a {fine} stays with you in later periods until the {ring} is detected or {reped}. A new partner "
    "means the old {ring} can no longer be punished.",
    "As long as the {ring} is still standing you risk a {fine}, also in periods in which you do not talk. Re-pairing "
    "cancels any risk from the old pairing.",
]

EARN = [
    "You start with 1000 points. Each period you earn your {profit} minus any {fine}, plus any {reward}; your total "
    "never falls below 0. {goal}",
    "Your starting balance is 1000 points; each period it changes by your {profit} less any {fine} plus any "
    "{reward}, and it cannot go below 0. {goal}",
    "Points: you begin with 1000; per period you gain {profit}, lose any {fine}, gain any {reward}; the balance is "
    "floored at 0. {goal}",
]
