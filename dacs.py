import logging
import math
from collections import OrderedDict
from random import Random

from .utils.lp import *


def create_training_selector(args):
    return _training_selector(args)


def create_testing_selector(data_distribution=None, client_info=None, model_size=None):
    return _testing_selector(data_distribution, client_info, model_size)


class _testing_selector:
    """Oort's testing selector

    We provide two kinds of selector:
    select_by_deviation: testing participant selection that preserves data representativeness.
    select_by_category: testing participant selection that enforce developer's requirement on
        distribution of the testing set. Note that this selector is avaliable only if the client
        info is provided.

    Attributes:
        client_info: Optional; A dictionary that stores client id to client profile(system speech and
            network bandwidth) mapping. For example, {1: [153.0, 2209.61]} indicates that client 1
            needs 153ms to run a single sample inference and their network bandwidth is 2209 Kbps.
        model_size: Optional; the size of the model(i.e., the data transfer size) in kb
        data_distribution: Optional; individual data characteristics(distribution).
    """

    def __init__(self, data_distribution=None, client_info=None, model_size=None):
        """Inits testing selector."""
        self.client_info = client_info
        self.model_size = model_size
        self.data_distribution = data_distribution
        if self.client_info:
            self.client_idx_list = list(range(len(client_info)))

    def update_client_info(self, client_ids, client_profile):
        """Update clients' profile(system speed and network bandwidth)

        Since the clients' info is dynamic, developers can use this function
        to update clients' profile. If the client id does not exist, Oort will
        create a new entry for this client.

        Args:
            client_ids: A list of client ids whose profile needs to be updated
            client_info: Updated information about client profile, formatted as
                a list of pairs(speed, bw)

        Raises:
            Raises an error if len(client_ids) != len(client_info)
        """
        return 0

    def _hoeffding_bound(self, dev_tolerance, capacity_range, total_num_clients, confidence=0.8):
        """Use hoeffding bound to cap the deviation from E[X]

        Args:
            dev_tolerance: maximum deviation from the empirical (E[X])
            capacity_range: the global max-min range of number of samples across all clients
            total_num_clients: total number of feasible clients
            confidence: Optional; Pr[|X - E[X]| < dev_tolerance] > confidence

        Returns:
            The estimated number of participant needed to satisfy developer's requirement
        """

        factor = (1.0 - 2 * total_num_clients / math.log(1 - math.pow(confidence, 1)) \
                  * (dev_tolerance / float(capacity_range)) ** 2)
        n = (total_num_clients + 1.0) / factor

        return n

    def select_by_deviation(self, dev_target, range_of_capacity, total_num_clients,
                            confidence=0.8, overcommit=1.1):
        """Testing selector that preserves data representativeness.

        Given the developer-specified tolerance `dev_target`, Oort can estimate the number
        of participants needed such that the deviation from the representative categorical
        distribution is bounded.

        Args:
            dev_target: developer-specified tolerance
            range_of_capacity: the global max-min range of number of samples across all clients
            confidence: Optional; Pr[|X - E[X]| < dev_tolerance] > confidence
            overcommit: Optional; to handle stragglers

        Returns:
            A list of selected participants
        """
        num_of_selected = self._hoeffding_bound(dev_target, range_of_capacity, total_num_clients, confidence=0.8)
        return num_of_selected

    def select_by_category(self, request_list, max_num_clients=None, greedy_heuristic=True):
        """Testing selection based on requested number of samples per category.

        When individual data characteristics(distribution) is provided, Oort can
        enforce client's request on the number of samples per category.

        Args:
            request_list: a list that specifies the desired number of samples per category.
                i.e., [num_requested_samples_class_x for class_x in request_list].
            max_num_clients: Optional; the maximum number of participants .
            greedy_heuristic: Optional; whether to use Oort-based solver. Otherwise, Mix-Integer Linear Programming
        Returns:
            A list of selected participants ids.

        Raises:
            Raises an error if 1) no client information is provided or 2) the requirement
            cannot be satisfied(e.g., max_num_clients too small).
        """
        client_sample_matrix, test_duration, lp_duration = run_select_by_category(request_list, self.data_distribution,
                                                                                  self.client_info, max_num_clients,
                                                                                  self.model_size, greedy_heuristic)
        return client_sample_matrix, test_duration, lp_duration


class _training_selector(object):
    """DACS (Deadline-Aware Client Selection) training selector.

    Implements Algorithm 1, ClientSelection(C, k, eps):
        Step 1: Utility(i) = U_stat(i) * TP(i) for every explored client i in E   (Eq. 6, Eq. 7, Eq. 9)
        Step 2: Exploitation; W <- top (1-eps)*k clients of E by Utility(i)
        Step 3: Preferred deadline T <- (3*T_min + T_max)/4                      (Eq. 11)
        Step 4: Exploration; add top eps*k clients of (C \\ E) by Utility(j)      (Eq. 6, Eq. 7, Eq. 9)
        Step 5: E <- E U W
    """

    def __init__(self, args, sample_seed=233):

        self.totalArms = OrderedDict()
        self.training_round = 0

        self.args = args
        self.epsilon = args.exploration_factor  # eps: exploitation/exploration split, 0 < eps < 1

        self.rng = Random()
        self.rng.seed(sample_seed)

        self.explored = set()  # E: set of explored clients
        self.round_prefer_duration = 0.0  # T: preferred deadline

        # sigma: bandwidth of the Gaussian temporal-proximity kernel used in TP(i), Eq. 7
        self.sigma = getattr(args, 'sigma', getattr(args, 'sigma0', 150))

    def register_client(self, client_id, feedbacks):
        # Initiate the score for arms. [score, time_stamp, # of trials, size of client, auxi, duration]
        if client_id not in self.totalArms:
            self.totalArms[client_id] = {}
            self.totalArms[client_id]['reward'] = feedbacks['reward']
            self.totalArms[client_id]['duration'] = feedbacks['duration']
            self.totalArms[client_id]['time_stamp'] = self.training_round
            self.totalArms[client_id]['count'] = 0
            self.totalArms[client_id]['status'] = True

    def update_client_util(self, client_id, feedbacks):
        '''
        @ feedbacks['reward']: statistical utility
        @ feedbacks['duration']: system utility
        @ feedbacks['count']: times of involved
        '''
        self.totalArms[client_id]['reward'] = feedbacks['reward']
        self.totalArms[client_id]['duration'] = feedbacks['duration']
        self.totalArms[client_id]['time_stamp'] = feedbacks['time_stamp']
        self.totalArms[client_id]['count'] += 1
        self.totalArms[client_id]['status'] = feedbacks['status']

    def select_participant(self, num_of_clients, feasible_clients=None):
        '''
        @ num_of_clients: # of clients selected
        '''
        viable_clients = feasible_clients if feasible_clients is not None else set(
            [x for x in self.totalArms.keys() if self.totalArms[x]['status']])
        return self.getTopK(num_of_clients, self.training_round + 1, viable_clients)

    def update_duration(self, client_id, duration):
        if client_id in self.totalArms:
            self.totalArms[client_id]['duration'] = duration

    def _statistical_utility(self, client_id):
        """U_stat(i): Eq. 6."""
        return self.totalArms[client_id]['reward']

    def _temporal_proximity(self, duration):
        """TP(i) = exp( -(t_i - T)^2 / (2*sigma^2) ): Eq. 7."""
        return math.exp(-((duration - self.round_prefer_duration) ** 2) / (2 * (self.sigma ** 2)))

    def _utility(self, client_id):
        """Utility(i) = U_stat(i) * TP(i): Eq. 9."""
        return self._statistical_utility(client_id) * self._temporal_proximity(self.totalArms[client_id]['duration'])

    def getTopK(self, numOfSamples, cur_time, feasible_clients):
        """Algorithm 1: ClientSelection(C, k, eps)."""
        self.training_round = cur_time

        C = [x for x in self.totalArms.keys() if int(x) in feasible_clients]

        # ---- Step 1: Utility(i) for every explored client i in E ----
        E = [x for x in C if x in self.explored]
        utility = {client_id: self._utility(client_id) for client_id in E}

        # ---- Step 2: Exploitation; W <- top (1-eps)*k clients of E ----
        explorationBudget = int(numOfSamples * self.epsilon)
        exploitLen = min(numOfSamples - explorationBudget, len(E))
        W = sorted(utility, key=utility.get, reverse=True)[:exploitLen]

        # ---- Step 3: Preferred deadline T <- (3*T_min + T_max)/4 ----
        if W:
            durations = sorted(self.totalArms[client_id]['duration'] for client_id in W)
            t_min, t_max = durations[0], durations[-1]
            self.round_prefer_duration = (3 * t_min + t_max) / 4

        # ---- Step 4: Exploration; add up to eps*k clients of (C \ E), backfilling
        # the shortfall from Step 2 when E has fewer than (1-eps)*k clients ----
        unexplored = [x for x in C if x not in self.explored]
        exploreLen = min(numOfSamples - len(W), len(unexplored))

        explore_utility = {client_id: self._utility(client_id) for client_id in unexplored}
        exploreClients = sorted(explore_utility, key=explore_utility.get, reverse=True)[:exploreLen]

        pickedClients = W + exploreClients

        # ---- Step 5: E <- E U W ----
        self.explored.update(pickedClients)

        logging.info(
            "Training selector: round {}, exploited {}, explored {}, total picked {}".format(
                cur_time, len(W), len(exploreClients), len(pickedClients)))

        return pickedClients

    def get_median_reward(self):
        feasible_rewards = [self.totalArms[x]['reward'] for x in self.totalArms.keys()]

        if len(feasible_rewards) > 0:
            return sum(feasible_rewards) / float(len(feasible_rewards))

        return 0

    def get_client_reward(self, armId):
        return self.totalArms[armId]

    def getAllMetrics(self):
        return self.totalArms
