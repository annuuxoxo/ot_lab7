"""Registry of the 37 algorithms of the Lab 7 catalogue."""
import importlib

CATALOGUE = [
    (1, "Weighted Sum", "alg01_weighted_sum", "classical"),
    (2, "epsilon-Constraint", "alg02_epsilon_constraint", "classical"),
    (3, "Benson's Method", "alg03_benson", "classical"),
    (4, "Weight Metric", "alg04_weight_metric", "classical"),
    (5, "Value Function", "alg05_value_function", "classical"),
    (6, "Goal Programming", "alg06_goal_programming", "classical"),
    (7, "Lexicographic", "alg07_lexicographic", "classical"),
    (8, "TOPSIS", "alg08_topsis", "ranking"),
    (9, "VIKOR", "alg09_vikor", "ranking"),
    (10, "VEGA", "alg10_vega", "non-elitist"),
    (11, "MOGA", "alg11_moga", "non-elitist"),
    (12, "NSGA", "alg12_nsga", "non-elitist"),
    (13, "NPGA", "alg13_npga", "non-elitist"),
    (14, "NSGA-II", "alg14_nsga2", "elitist"),
    (15, "NSGA-III", "alg15_nsga3", "many-objective"),
    (16, "SPEA2", "alg16_spea2", "elitist"),
    (17, "PESA-II", "alg17_pesa2", "elitist"),
    (18, "MOEA/D", "alg18_moead", "decomposition"),
    (19, "MOEA/D-DE", "alg19_moead_de", "decomposition"),
    (20, "MOEA/D-ACO", "alg20_moead_aco", "decomposition"),
    (21, "Constrained NSGA-II", "alg21_constrained_nsga2", "constrained"),
    (22, "MOPSO", "alg22_mopso", "swarm"),
    (23, "MOGWO", "alg23_mogwo", "swarm"),
    (24, "MOABC", "alg24_moabc", "swarm"),
    (25, "MOACO", "alg25_moaco", "swarm"),
    (26, "MOSA", "alg26_mosa", "physics"),
    (27, "MODE", "alg27_mode", "evolutionary"),
    (28, "MO-TLBO", "alg28_motlbo", "human-inspired"),
    (29, "MOBA", "alg29_moba", "swarm"),
    (30, "MOMA", "alg30_moma", "hybrid"),
    (31, "RVEA", "alg31_rvea", "many-objective"),
    (32, "IBEA", "alg32_ibea", "indicator"),
    (33, "SMS-EMOA", "alg33_smsemoa", "indicator"),
    (34, "HypE", "alg34_hype", "indicator"),
    (35, "R-NSGA-II", "alg35_rnsga2", "preference"),
    (36, "MGDA", "alg36_mgda", "gradient"),
    (37, "GradNorm", "alg37_gradnorm", "gradient"),
]
NAME = {n: nm for n, nm, _, _ in CATALOGUE}


def load(n):
    mod = [m for k, _, m, _ in CATALOGUE if k == n][0]
    return importlib.import_module("algorithms." + mod)
