"""Wording O, paraphrase 0: the Leniency instructions of Bigoni et al. (SSE/EFI WP 698, Appendix 1; verbatim text in
notes/bigoni_instructions.md section 3) adapted per regime with minimal edits. The edit list lives in the comment
block of bigoni_prompts.py. "{TABLE}" marks where the profit table goes; {F} {F2} {A} are filled by the caller.
"""

from bigoni_wording import GOAL

RING = "price-fixing ring"   # replaces the original's word for the illegal arrangement (forbidden in this project)

INTRO = (
    "Welcome to this experiment about decision making in a market.\n\n"
    "In summary, the situation you will face is the following. You and one other participant referred to as your "
    "competitor produce similar goods and sell them in a common market. As in most markets, the higher the price you "
    "charge, the more you earn on each sold good, but the fewer goods you sell. And, as in many markets, the lower the "
    "price charged by your competitor, the more customers he or she will take away from you and the less you will "
    "sell and earn. It is possible, however, to form a " + RING + " with your competitor, that is, you will have the "
    "possibility to communicate and try to agree on prices at which to sell the goods.")

ILLEGAL = (
    " In reality, " + RING + "s are illegal and if the government discovers the " + RING + ", " + RING + " members are "
    "fined. In addition members of a " + RING + " can always report it to the government. The same happens in this "
    "experiment. If you communicate to discuss prices, even if both of you do not report, there is still a chance "
    "that the 'government' discovers it and if this happens, you will have to pay a 'fine'.")
LEGAL_NOTE = (" In this market communicating to discuss prices is legal: there are no fines and nobody can report "
              "anything.")
# The original sentence on payoffs of reports; one variant per regime kind.
INTRO_REPORT = {
    "L": (" If you report, and if you are the only one to report, you will not pay any fine but your competitor will "
          "pay the full fine. Conversely, if only your competitor reports the " + RING + ", you will pay the full fine "
          "and your competitor will not pay any fine. If instead both of you report the " + RING + " you will both "
          "pay 50% of the fine."),
    "F": (" If either of you reports the " + RING + ", you will both pay the full fine, including the one who "
          "reports."),
    "R": (" If you report, and if you are the only one to report, you will not pay any fine and you will receive the "
          "full fine as a reward, while your competitor will pay the full fine. Conversely, if only your competitor "
          "reports the " + RING + ", you will pay the full fine and your competitor will not pay any fine and will "
          "receive the reward. If instead both of you report the " + RING + " you will both pay 50% of the fine."),
}
INTRO_REPORT["T"] = INTRO_REPORT["F"]

TIMING = (
    "Timing of the experiment\n"
    "In this experiment you will be asked to make decisions in several periods. You will be paired with another "
    "participant for a sequence of periods. Such a sequence of periods is referred to as a match. You will never know "
    "with whom you have been matched in this experiment.\n"
    "The length of a match is random. After each period, there is a probability of 85% that the match will continue "
    "for at least another period. So, for instance, if you have been paired with the same competitor for 2 periods, "
    "the probability that you will be paired with him or her a third period is 85%. If you have been paired with the "
    "same competitor for 9 periods, the probability that you will be paired with him or her a tenth period is also "
    "85%.\n"
    "Once a match ends, you will be paired with another participant for a new match, unless 20 periods or more have "
    "passed. In this case the experiment ends. So, for instance, if 19 periods have passed, with a probability of 15% "
    "you are re-matched, that is you are paired with another participant. If 21 periods have passed, with a "
    "probability of 15% the experiment ends. You are never re-matched with the competitor you had in the period "
    "just before.")
TIMING_LAPSE = (" When you are re-matched you cannot be fined anymore for a " + RING + " formed in your previous "
                "match with your previous competitor.")

PRICES = (
    "Prices and Profits\n"
    "In each period you choose the price of your product. Your price as well as the price chosen by your competitor "
    "determines the quantity that you will sell.\n"
    "The higher your price, the more you earn on each sold good, but the fewer goods you sell. Therefore your price "
    "has two opposing effects on your profit. On the one hand, an increase in your price may increase your profit, "
    "since each good that you sell will earn you more money. On the other hand, an increase in your price may "
    "decrease your profit, since you will sell less.\n"
    "Furthermore, the higher the price of your competitor, the more you will sell. As a result, your profits increase "
    "if your competitor chooses a higher price.\n"
    "To make things easy, we have constructed a profit table. This table is added to the instructions. Have a look at "
    "this table now. Your own prices are indicated next to the rows and the prices of your competitor are indicated "
    "above the columns. If you want to know your profit if, for example, your competitor's price is 5 and your price "
    "is 4, then you first move to the right until you find the column with 5 above it, and then you move down until "
    "you reach the row which has 4 on the left of it. You can read that your profit is 160 points in that case.\n"
    "Your competitor has received an identical table. Therefore you can also use the table to learn your "
    "competitor's profit by inverting your roles. That is, read the price of your competitor next to the rows and "
    "your price above the columns. In the previous example where your price is equal to 4 and your competitor's "
    "price is equal to 5, it follows that your competitor's profit is 100 points.\n"
    "Note that if your and your competitor's prices are equal, then your profits are also equal and are indicated in "
    "one of the cells along the table's diagonal. For example, if your price and the price of your competitor are "
    "equal to 1, then your profit and the profit of your competitor is equal to 38 points. If both you and your "
    "competitor increase your price by 1 point to 2, then your profit and the profit of your competitor becomes "
    "equal to 71.\n"
    "Note also that if your competitor's price is sufficiently low relative to your price, then your profit is equal "
    "to 0. The reason is that no consumer buys your good, since it is too expensive relative to your competitor's "
    "good.\n\n{TABLE}")

FINES = {
    "L": ("Fines\n"
          "In every period, you and your competitor will be given the opportunity to communicate and discuss prices. If "
          "both of you agree to communicate, you will be considered to have formed a " + RING + ", and then you might "
          "have to pay a fine F. This fine is given by:\nF = {F} points\n"
          "You can be fined in two ways. First, you and your competitor will have the opportunity to report the "
          + RING + ". If you are the only one to report the " + RING + ", you will not pay any fine but your "
          "competitor will pay the full fine, that is {F} points. Conversely, if only your competitor reports the "
          + RING + " and you do not, then you will have to pay the full fine equal to {F} points and your competitor "
          "will not pay any fine. Finally, if both of you report the " + RING + ", you will both pay 50% of the fine, "
          "that is {F2} points.\n"),
    "F": ("Fines\n"
          "In every period, you and your competitor will be given the opportunity to communicate and discuss prices. If "
          "both of you agree to communicate, you will be considered to have formed a " + RING + ", and then you might "
          "have to pay a fine F. This fine is given by:\nF = {F} points\n"
          "You can be fined in two ways. First, you and your competitor will have the opportunity to report the "
          + RING + ". If either of you reports the " + RING + ", you will both pay the full fine, that is {F} points, "
          "including the one who reports.\n"),
    "R": ("Fines\n"
          "In every period, you and your competitor will be given the opportunity to communicate and discuss prices. If "
          "both of you agree to communicate, you will be considered to have formed a " + RING + ", and then you might "
          "have to pay a fine F. This fine is given by:\nF = {F} points\n"
          "You can be fined in two ways. First, you and your competitor will have the opportunity to report the "
          + RING + ". If you are the only one to report the " + RING + ", you will not pay any fine and you will "
          "receive a reward equal to the full fine, that is {F} points, while your competitor will pay the full fine. "
          "Conversely, if only your competitor reports the " + RING + " and you do not, then you will have to pay the "
          "full fine equal to {F} points and your competitor will not pay any fine and will receive the reward. "
          "Finally, if both of you report the " + RING + ", you will both pay 50% of the fine, that is {F2} points.\n"),
}
FINES["T"] = FINES["F"]
FINES_TAIL = (
    "Second, if neither you nor your competitor reports the " + RING + ", the government discovers it with the "
    "following probability.\nProbability of detection = {A}%.\n"
    "Note that you will run the risk of paying a fine as long as the " + RING + " has not yet been discovered or "
    "reported. Thus you may pay a fine in a period even if no meeting takes place in that period. This happens if you "
    "had a meeting in some previous period which has not yet been discovered or reported.\n"
    "Once a " + RING + " is discovered or reported, you do not anymore run the risk of paying a fine in future "
    "periods, unless you and your competitor agree to communicate again.")

EARN = {
    "C": "Earnings\nThe number of points you earn in a period will be equal to your profit.",
    "R": ("Earnings\nThe number of points you earn in a period will be equal to your profit minus an eventual fine or "
          "plus an eventual reward. Note that because of the fine, your earnings may be negative in some periods. "
          "Your cumulated earnings, however, will never be allowed to become negative."),
    "other": ("Earnings\nThe number of points you earn in a period will be equal to your profit minus an eventual "
              "fine. Note that because of the fine, your earnings may be negative in some periods. Your cumulated "
              "earnings, however, will never be allowed to become negative."),
}
ENDOW = ("You will receive an initial endowment of 1000 points and, as the experiment proceeds, your and your "
         "competitor's decisions will determine your cumulated earnings. " + GOAL)

STEPS_HEAD = ("Decision making in a period\nNext we describe in more detail how you make decisions in each period. A "
              "period is divided into {n} steps. Some steps will inform you about decisions that you and your "
              "competitor have made. In the other steps you and your competitor will have to make decisions. If your "
              "reply to a decision cannot be read, the computer will make a decision for you: you will not "
              "communicate, you will not report, and your price will be chosen so high that your profits will be 0.")
STEP1 = (
    "Step 1: Pairing information and price communication decision\n"
    "Every period starts by informing you whether or not you will play against the same competitor as in the "
    "previous period.\n{remember}"
    "In this step you will also be asked if you want to communicate with your competitor to discuss prices. A "
    "communication screen will open only if BOTH you and your competitor answer YES. Otherwise there is no "
    "communication and the pricing decision follows in Step 3.")
REMEMBER = ("Remember that if you are paired with a new competitor, you cannot be fined anymore for " + RING
            + "s that you formed with your previous competitors.\n")
STEP2 = (
    "Step 2: Price communication\n"
    "After the communication screen has opened, you can \"discuss\" prices by choosing a price out of the range "
    "{{ 0, 1, 2, . . . , 12 }}. In this way you can indicate to your competitor the minimum price that you find "
    "acceptable for both of you. When both of you have chosen a price, these two prices are displayed. You can then "
    "choose a new price but now this price should be greater or equal to the smaller of the two previously chosen "
    "prices. This procedure is repeated for 3 rounds. The smaller of the two last chosen prices is then displayed, "
    "which is referred to as the agreed-upon price. Note, however, that in the next step, neither you nor your "
    "competitor is forced to choose the agreed-upon price.")
STEP3 = (
    "Step 3: Pricing decision\n"
    "You and your competitor must choose one of the following prices: 0, 1, 2, . . . , 12. When you choose your "
    "price, your competitor will not observe your choice nor will you observe his or her price choice. This "
    "information is only revealed in Step {reveal}.")
STEP3_REPORT = ("\nThe experiment proceeds to the first reporting decision in Step 4 if you communicated in Step 2 or if "
                "in previous periods you formed a " + RING + " not yet discovered or reported.")
STEP4 = {
    "L": "As described above, if you are the only one to report, you will not pay the fine; the opposite happens if "
         "only your competitor reports; and if both of you report, you will both pay 50% of the fine.",
    "F": "As described above, if either of you reports, you will both pay the full fine.",
    "R": "As described above, if you are the only one to report, you will not pay the fine and you will receive the "
         "reward; if only your competitor reports, you will pay the fine and your competitor receives the reward; and "
         "if both of you report, you will both pay 50% of the fine.",
}
STEP4["T"] = STEP4["F"]
STEP4_TXT = (
    "Step 4: First (secret) reporting decision\n"
    "By answering \"yes\" to the reporting question, you report that you have been communicating in the past. "
    "{payoff}\n"
    "If you do not wish to report, answer \"no\" instead.\n"
    "When you decide whether or not to report, your competitor will not observe your choice, nor will you observe "
    "his or her choice. This information is only revealed when market prices are revealed in Step 5.")
STEP5 = {
    "L": "Again, if you are the only one to report, you will not pay the fine. On the contrary, if your competitor "
         "reports and you don't you will have to pay the fine and he will not. If both you and your competitor "
         "report, you will both pay 50% of the fine, that is {F2} points.",
    "F": "Again, if either of you reports, you will both pay the full fine, that is {F} points.",
    "R": "Again, if you are the only one to report, you will not pay the fine and you will receive the reward. On the "
         "contrary, if your competitor reports and you don't you will have to pay the fine and he will receive the "
         "reward. If both you and your competitor report, you will both pay 50% of the fine, that is {F2} points.",
}
STEP5["T"] = STEP5["F"]
STEP5_TXT = (
    "Step 5: Market prices and second reporting decision\n"
    "In this step your and your competitor's prices and profits are displayed.\n"
    "In case you have formed a " + RING + " not yet discovered or reported, the screen will also display whether or "
    "not you or your competitor reported it in the first reporting step (Step 4). If not, you will get a new "
    "opportunity to report.\n"
    "If you wish to report, answer \"yes\". If you do not wish to report, answer \"no\".\n"
    "{payoff}")
STEP6 = (
    "Step 6: Detection probability\n"
    "If this step is reached, you formed a " + RING + " either in the current period or in previous periods. "
    "Furthermore the " + RING + " has not yet been discovered or reported. The " + RING + " can nevertheless be "
    "discovered. This happens with a probability of {A}%. If the " + RING + " is discovered, you and your "
    "competitor will have to pay the full fine of {F} points.")
STEP7 = (
    "Step {n}: Summary\n"
    "In this step you learn the choices made in the previous steps: your and your competitor's price choices and "
    "profits, {fine_part}and your earnings.{rep_part}{end_part}")
ENDING = (
    "Period ending and ending of the experimental session\n"
    "After the summary, a new period starts unless 20 or more periods have passed and the 15% probability of pair "
    "dismantling takes place. In that case, the experiment ends.\n"
    "Throughout the experiment, a table will keep track for you of the history with your current competitor. For each "
    "previous period played with your current competitor, this table will show your price and profit, your "
    "competitor's price and profit{fine_hist}.")
