from scipy.stats import beta as beta_dist
import random
random.seed()

class Candidate:
    def __init__(self, name, stance, popularity):
        self.name = name
        self.stance = stance  # 0–100 scale
        self.popularity = popularity
        self.votes_ranked = 0.0
        self.votes_unranked = 0
        self.votes_one_choice = 0

    def __repr__(self):
        return f"{self.name} ({self.stance})"
    
class Voter:
    def __init__(self, political_score):
        """
        Initialize a voter with a political score (1-100).
        """
        self.political_score = political_score
        self.compatibilities = {}  # Will store {candidate_name: (distance, candidate_popularity)}

    def evaluate_candidates(self, candidates):
        """
        Given a list of Candidate objects, calculate compatibility with each.
        Compatibility is based on distance in political views.
        """
        self.compatibilities = {}
        for candidate in candidates:
            distance = abs(self.political_score - candidate.stance)
            self.compatibilities[candidate.name] = {
                "distance": distance,
                "popularity": candidate.popularity
            }

def solve_beta_parameter(fixed_value, is_alpha_fixed, target_cdf, tolerance=1e-6, max_iter=100):
    """
    Binary search to find the free parameter (alpha or beta) such that the CDF at 0.5
    matches a desired target value (i.e., 1 - popularity). One of alpha or beta is fixed at 1.0.
    The function dynamically adjusts the other parameter to skew the Beta distribution left or right.
    Direction of adjustment depends on which parameter is fixed:
    - If alpha is fixed, increasing beta shifts mass left (CDF ↑)
    - If beta is fixed, increasing alpha shifts mass right (CDF ↓)
    Returns (alpha, beta) pair that achieves the target CDF(0.5) within tolerance.
    """
    low = 0.01
    high = 100  # allow full range, catch over-polarization later
    for _ in range(max_iter):
        free = (low + high) / 2

        if is_alpha_fixed:
            cdf = beta_dist.cdf(0.5, fixed_value, free)

            # We are solving for beta:
            # ↑ increase beta → more mass on left → CDF ↑
            if cdf > target_cdf:
                high = free
            else:
                low = free

        else:
            cdf = beta_dist.cdf(0.5, free, fixed_value)

            # We are solving for alpha:
            # ↑ increase alpha → more mass on right → CDF ↓
            if cdf < target_cdf:
                high = free
            else:
                low = free

        if abs(cdf - target_cdf) < tolerance:
            return (fixed_value, free) if is_alpha_fixed else (free, fixed_value)

    return (fixed_value, (low + high) / 2) if is_alpha_fixed else ((low + high) / 2, fixed_value)

def find_beta_from_popularity(popularity, tolerance=1e-6):
    """
    Given a right-wing alignment rate (percent > 0.5), return (α, β) so that:
        CDF(0.5) = 1 - popularity
    One of α or β is fixed at 1; other is solved.
    """
    assert 0 < popularity < 1, "right_alignment_rate must be strictly between 0 and 1"
    target_cdf = 1 - popularity

    if abs(target_cdf - 0.5) < tolerance:
        return 1.0, 1.0  # uniform

    if popularity > 0.5:
        return solve_beta_parameter(1.0, is_alpha_fixed=True, target_cdf=target_cdf, tolerance=tolerance)
    else:
        return solve_beta_parameter(1.0, is_alpha_fixed=False, target_cdf=target_cdf, tolerance=tolerance)

def get_alpha_and_beta():
    """
    This function's main purpose is to obtain a value for alpha and beta, along with the rate that people lean politically right. 
    Much of this is chosen by the user. 
    """

    print("""
-------------------------------
CANDIDATE OPINION DISTRIBUTION
-------------------------------

This model uses the Beta distribution to simulate how the population feels politically.
The Beta distribution allows us to represent everything from widespread support
to deeply polarized opinions.

--- Right Wing Alignment Rate ---
You must provide a 'right wing alignment rate' value between 0 and 1.
This represents the percentage of voters who lean politically right. 
(For example, right wing alignment rate = 0.7 means 70% of voters lean politically right.)

--- DISTRIBUTION SHAPE ---
By default, the distribution is shaped based on the right wing alignment rate:
- If right wing alignment rate > 0.5 → α = 1, solve for β
- If right wing alignment rate < 0.5 → β = 1, solve for α
- If right wing alignment rate = 0.5 → uniform distribution (α = β = 1)

This keeps the model simple and realistic without overcomplicating it.

--- ADVANCED MODE (OPTIONAL) ---
If you prefer full control, you can specify your own values for α and β.
Alternatively, you can specify the mean (µ) and standard deviation(σ).
Use the Desmos graph below to explore how different values affect the distribution:

→ https://www.desmos.com/calculator/2rg2rlsu1u

- α < 1 skews toward low scores (disapproval)
- β < 1 skews toward high scores (approval)
- Both < 1 = polarized (extremes)
- Both > 1 = moderate consensus

Only one input mode should be used at a time.
Type "default" or "advanced" to continue. 

-------------------------------
""")
    user_input = input("> ").strip().lower()

    while (user_input != "default") and (user_input != "advanced"):
        print("Not a valid option. ")
        user_input = input("> ").strip().lower()

    if user_input == "default":
        popularity_works = False
        while (popularity_works == False) or (popularity <= 0) or (popularity >= 1):
            try:
                popularity = 1 - float(input("What  percent of the population leans right politically? "))  # 70% rate above 0.5 ⇒ CDF(0.5) = 0.3
                popularity_works = True
                if (popularity <= 0) or (popularity >= 1):
                    print("Right-wing alignment rate must be on the open interval (0,1). ")
            except:
                print("Not a valid input for right wing alignment rate. ")

        alpha_param, beta_param = find_beta_from_popularity(popularity)

    elif user_input == "advanced":
        print("https://www.desmos.com/calculator/2rg2rlsu1u")
        print()
        alpha_param_works = False
        while alpha_param_works == False or (alpha_param <= 0) or (alpha_param > 50):
            try:
                alpha_param = float(input("What is ⍺? "))
                alpha_param_works = True
                if (alpha_param <= 0) or (alpha_param > 50):
                    print("⍺ must be more than 0 and less than 50. ")
            except:
                print("Not a valid input for ⍺. ")

        beta_param_works = False
        while beta_param_works == False or (beta_param <= 0) or (beta_param > 50):
            try:
                beta_param = float(input("What is β? "))
                beta_param_works = True
                if (beta_param <= 0) or (beta_param > 50):
                    print("β must be more than 0 and less than 50. ")
            except:
                print("Not a valid input for β. ")



    cdf = beta_dist.cdf(0.5, alpha_param, beta_param)
    print(f"α = {alpha_param:.6f}, β = {beta_param:.6f}")
    print(f"CDF(0.5) = {cdf:.6f} → right alignment rate = {1-cdf:.6f}\n")

    return (alpha_param,beta_param)

def biased_random(candidate_parameters):
    """
    This fuction generates a random number between 1 and 100, representing a 
    voter's views on the political spectrum. 50 or below is left, 51 or above is right. 
    Input: parameters is a tuple that contains a value for alpha and beta, 
           along with the rate that the population leans right, but the that is not used here
    """
    return int(random.betavariate(candidate_parameters[0],candidate_parameters[1])*100)

def candidate_popularity(candidate_views,alpha_param,beta_param):
    if candidate_views >= 0.8:
        return (1 - (beta_dist.cdf(candidate_views - 0.2, alpha_param, beta_param)))
    elif candidate_views <= 0.2:
        return beta_dist.cdf(candidate_views + 0.2, alpha_param, beta_param)
    else:
        return ((beta_dist.cdf(candidate_views + 0.2, alpha_param, beta_param)) - (beta_dist.cdf(candidate_views - 0.2, alpha_param, beta_param)))

def political_landscape():
    # User defines ideological landscape via Beta distribution
    print()
    distribution = get_alpha_and_beta()

    # Ask for candidate info
    print("\nCandidate Setup")
    while True:
        try:
            num = int(input("How many candidates? "))
            if num < 1 or num > 7:
                print("Please choose between 1 and 7 candidates.")
                continue
            break
        except:
            print("Invalid number.")

    candidates = []
    for i in range(num):
        name = chr(97 + i)  # a, b, c, ...
        while True:
            try:
                print("Values closer to 1 are left leaning, values closser to 100 are right leaning. ")
                stance = int(input(f"Enter political stance (1-100) for candidate {name}: "))
                if stance < 1 or stance > 100:
                    print("Stance must be between 1 and 100.")
                    continue
                break
            except:
                print("Invalid stance.")
        popularity = candidate_popularity(stance/100,distribution[0],distribution[1])
        candidates.append(Candidate(name, stance, popularity))
    

    # Ask for population size
    population_works = False
    while population_works == False or (population <= 0) or (population > 100000):
        try:
            population = int(input("Enter the desired population size: "))
            population_works = True
            if (population <= 0) or (population > 100000):
                print("The population size must be more than 0 and less than 100000. ")
        except:
            print("Not a valid input for the population size. ")

    return distribution, population, candidates

def run_election(simulation_parameters):

    distribution = simulation_parameters[0]
    population = simulation_parameters[1]
    candidates = simulation_parameters[2]
    one_choice_votes = []
    ranked_votes = []
    unranked_votes = []

    
    for i in range(population):

        # Determines how much a voter likes a candidate
        ballot = []
        eligible_voter = Voter(biased_random(distribution))
        Voter.evaluate_candidates(eligible_voter, candidates)


        # Determines who a voter votes for in the ranked system
        ballot = []
        for i in eligible_voter.compatibilities:
            opinion = [i, eligible_voter.compatibilities[i]["distance"], eligible_voter.compatibilities[i]["popularity"]]
            ballot.append(opinion)
        for i in range(len(ballot)):
            for j in range(0, len(ballot) - i - 1):
                if ballot[j][1] > ballot[j + 1][1]:
                    ballot[j], ballot[j + 1] = ballot[j + 1], ballot[j]
        filled_out_ballot = []
        for i in range(len(ballot)):
            filled_out_ballot.append(ballot[i][0])

        ranked_votes.append(filled_out_ballot)


        # Determines who a voter votes for in the unranked system
        blank_ballot = []
        for i in eligible_voter.compatibilities:
            opinion = [i,eligible_voter.compatibilities[i]["distance"],eligible_voter.compatibilities[i]["popularity"]]
            blank_ballot.append(opinion)
        ballot = []
        for i in range(len(blank_ballot)):
            if blank_ballot[i][1] < 20:
                ballot.append(blank_ballot[i])
        filled_out_ballot = []
        for i in range(len(ballot)):
            filled_out_ballot.append(ballot[i][0])
        
        unranked_votes.append(filled_out_ballot)

        # Determines who a voter votes for a candidate in the one-choice system
        blank_ballot = []
        for i in eligible_voter.compatibilities:
            opinion = [i,eligible_voter.compatibilities[i]["distance"],eligible_voter.compatibilities[i]["popularity"]]
            blank_ballot.append(opinion)
        ballot = []
        for i in range(len(blank_ballot)):
            if blank_ballot[i][1] < 20:
                ballot.append(blank_ballot[i])
        for i in range(len(ballot)):
            for j in range(0, len(ballot) - i - 1):
                if ballot[j][2] > ballot[j + 1][2]:
                    ballot[j], ballot[j + 1] = ballot[j + 1], ballot[j]
        filled_out_ballot = []
        for i in range(len(ballot)):
            filled_out_ballot.append(ballot[i][0])

        one_choice_votes.append(filled_out_ballot)


    # Counts votes for ranked system
    for vote in ranked_votes:  # each vote is a list like ['b', 'c', 'a', 'd']
        for i, candidate_name in enumerate(vote):  # i = rank (0-indexed)
            for candidate in candidates:
                if candidate.name == candidate_name:
                    candidate.votes_ranked += (len(candidates) - i) / len(candidates)


    # Counts votes for unranked system
    for i in range(len(unranked_votes)):
        for j in candidates:
            if j.name in unranked_votes[i]:
                j.votes_unranked += 1


    # Counts votes for one choice system
    for i in range(len(one_choice_votes)):
        for j in candidates:
            try:
                if one_choice_votes[i][0] == j.name:
                    j.votes_one_choice += 1
            except:
                pass

    # Determine ranked election winner
    ranked_candidates = sorted(candidates, key=lambda c: c.votes_ranked, reverse=True)
    ranked_election_winner = ranked_candidates[0].name     # always d, wrong

    # Determine one choice election winner
    ranked_candidates = sorted(candidates, key=lambda c: c.votes_one_choice, reverse=True)
    one_coice_election_winner = ranked_candidates[0].name 


    # Determine unranked  election winner
    ranked_candidates = sorted(candidates, key=lambda c: c.votes_unranked, reverse=True)
    unranked_election_winner = ranked_candidates[0].name 


    # Print election results
    print(f"""
------------------------------
One-Choice Election Results!!!
------------------------------
Candidate {one_coice_election_winner} won the election!

      
Population: {population}
""")
    
    for i in candidates:
        print(f"{i.name}'s # of votes: {i.votes_one_choice} votes")
    print()
    for i in candidates:
        print(f"{i.name}'s % of votes: {(i.votes_one_choice/population)*100} %")
    print()
    for i in candidates:
        print(f"{i.name}'s popularity: {i.popularity*100} %")
    print()


    print(f"""
--------------------------
Ranked Election Results!!!
--------------------------
Candidate {ranked_election_winner} won the election!        

Population: {population}
""")

    for i in candidates:
        print(f"{i.name}'s # of votes: {i.votes_ranked} votes")     # is not sorted
    print()
    for i in candidates:
        print(f"{i.name}'s % of votes: {(i.votes_ranked/population)*100} %")     # is not sorted
    print()
    for i in candidates:
        print(f"{i.name}'s popularity: {i.popularity*100} %")
    print()

    print(f"""
----------------------------
Unranked Election Results!!!
----------------------------
Candidate {unranked_election_winner} won the election!

      
Population: {population}
""")

    for i in candidates:
        print(f"{i.name}'s # of votes: {i.votes_unranked} votes")
    print()
    for i in candidates:
        print(f"{i.name}'s % of votes: {(i.votes_unranked/population)*100} %")
    print()
    for i in candidates:
        print(f"{i.name}'s popularity: {i.popularity*100} %")
    print()

def print_info():
    print("""
╔════════════════════════════════════════════════════════════════╗
║                 PUBLIC OPINION ELECTION SIMULATOR              ║
╚════════════════════════════════════════════════════════════════╝

This simulator models how a population votes for a custom number of candidates,
based on how politically aligned voters feel with each one. Voter opinions are
generated using a flexible probability model built from the Beta distribution.

═══════════════════════════════════════════════════════════════════════
HOW IT WORKS
═══════════════════════════════════════════════════════════════════════

→ Voters each receive a random "political alignment score" from 1 to 100
   based on the Beta distribution you configure.

→ Candidates are each given:
    • A name (like a, b, c, etc.)
    • A political stance on the same 1–100 scale

→ Each voter evaluates candidates based on:
    • Ideological **distance** from their own views
    • Candidate's **overall popularity**, estimated from the distribution

═══════════════════════════════════════════════════════════════════════
ELECTION SYSTEMS SIMULATED
═══════════════════════════════════════════════════════════════════════

1.   **One-Choice Voting**  
   Voters pick only one candidate: the one that's both close in ideology
   and broadly popular.

2.  **Ranked Voting**  
   Voters rank all candidates from most to least compatible.
   Points are distributed as:
       - 1st place → N/N points
       - 2nd place → (N-1)/N points
       - ...
       - Last place → 1/N points

3.  **Unranked Approval Voting**  
   Voters "approve" of any candidate whose ideology is close enough to
   their own (within 20 points).

═══════════════════════════════════════════════════════════════════════
HOW TO DEFINE PUBLIC OPINION
═══════════════════════════════════════════════════════════════════════

→ **default**: You input how many voters lean right (e.g. 70%), and the simulator
   automatically chooses the distribution shape for realism.

→ **advanced**: You manually enter Alpha (α) and Beta (β) values to shape the
   political distribution however you like.
   Use this Desmos tool to explore:
   🔗 https://www.desmos.com/calculator/2rg2rlsu1u

Beta Distribution Tips:
   - α < 1  → most voters disapprove
   - β < 1  → most voters approve
   - both < 1 → polarized extremes (love/hate)
   - both > 1 → mild, centrist consensus

I highly recommend looking at these two research papers:
    🔗 https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2830372
    🔗 https://www.researchgate.net/publication/331008377_Opinion_Polarization_during_a_Dichotomous_Electoral_Process

          
Along with this video by Veritasium:
    🔗 https://www.youtube.com/watch?v=qf7ws2DF-zk
          
═══════════════════════════════════════════════════════════════════════
GOAL OF THE SIMULATION
═══════════════════════════════════════════════════════════════════════

Can a broadly liked candidate win under all systems?  
Or will voting rules reward fringe appeal or strategic compromise?

Try different settings and see how public perception shapes electoral outcomes!

""")
    input("Press Enter to continue: ")

def print_menu():
    print('''
═══════════════════════════════════════════════════════════════════════
  MENU COMMANDS
═══════════════════════════════════════════════════════════════════════
After you set the candidates' statistics, you can:

• "politics"     → Redefine public opinion
• "simulate"     → Simulate and display all three voting systems
• "info"         → Print the guide again
• "seed"         → Change the seed for the random number generator
• "quit"         → Exit the program
''')

print_info()
simulation_parameters = political_landscape()
command = ""
while (command != "quit") and (command != "exit"):
    print_menu()
    command = input("> ").strip().lower()
    if command == "politics":
        simulation_parameters = political_landscape()
    elif command == "simulate":
        run_election(simulation_parameters)
    elif command == "info":
        print_info()
    elif command == "seed":
        random.seed(input("Enter a seed: "))
    elif (command == "quit") or (command == "exit"):
        print("Goodbye!")
    else:
        print("Not a valid command. ")

