"""
Training a linear controller on latent + recurrent state
with CMAES.

This is a bit complex. num_workers slave threads are launched
to process a queue filled with parameters to be evaluated.
"""
import argparse
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from time import sleep
from torch.multiprocessing import Process, Queue
import torch
import cma
from models import Controller
from tqdm import tqdm
import numpy as np
from utils.misc import RolloutGenerator, RolloutGeneratorVAERNN, ASIZE, RSIZE, LSIZE
from utils.misc import load_parameters
from utils.misc import flatten_parameters
import pickle

# parsing
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where everything is stored.')
parser.add_argument('--n-samples', type=int, help='Number of samples used to obtain '
                    'return estimate.')
parser.add_argument('--pop-size', type=int, help='Population size.')
parser.add_argument('--target-return', type=float, help='Stops once the return '
                    'gets above target_return')
parser.add_argument('--display', action='store_true', help="Use progress bars if "
                    "specified.")
parser.add_argument('--max-workers', type=int, help='Maximum number of workers.',
                    default=30)
args = parser.parse_args()

# Max number of workers. M

# multiprocessing variables
n_samples = args.n_samples
pop_size = args.pop_size
num_workers = min(args.max_workers, n_samples * pop_size)
time_limit = 1000

# create tmp dir if non existent and clean it if existent
tmp_dir = join(args.logdir, 'tmp')
if not exists(tmp_dir):
    mkdir(tmp_dir)
else:
    for fname in listdir(tmp_dir):
        unlink(join(tmp_dir, fname))

# create ctrl dir if non exitent
# ctrl_dir = join(args.logdir, 'ctrl_vaernn')
ctrl_dir = join(args.logdir, 'ctrl_vae_vanilla_pop64_NEW')
if not exists(ctrl_dir):
    mkdir(ctrl_dir)


################################################################################
#                           Thread routines                                    #
################################################################################
def slave_routine(p_queue, r_queue, e_queue, p_index):
    print('went into slave_routine!') 
    """ Thread routine.

    Threads interact with p_queue, the parameters queue, r_queue, the result
    queue and e_queue the end queue. They pull parameters from p_queue, execute
    the corresponding rollout, then place the result in r_queue.

    Each parameter has its own unique id. Parameters are pulled as tuples
    (s_id, params) and results are pushed as (s_id, result).  The same
    parameter can appear multiple times in p_queue, displaying the same id
    each time.

    As soon as e_queue is non empty, the thread terminate.

    When multiple gpus are involved, the assigned gpu is determined by the
    process index p_index (gpu = p_index % n_gpus).

    :args p_queue: queue containing couples (s_id, parameters) to evaluate
    :args r_queue: where to place results (s_id, results)
    :args e_queue: as soon as not empty, terminate
    :args p_index: the process index
    """
    # init routine
    gpu = p_index % torch.cuda.device_count()
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    
    print('gpu selected!') 

    # redirect streams
    sys.stdout = open(join(tmp_dir, str(getpid()) + '.out'), 'a')
    sys.stderr = open(join(tmp_dir, str(getpid()) + '.err'), 'a')
    
    print('streams redirected') 

    with torch.no_grad():
        r_gen = RolloutGenerator(args.logdir, device, time_limit)
        #r_gen = RolloutGeneratorVAERNN(args.logdir, device, time_limit)
        #print("got rgen")

        while e_queue.empty():
            if p_queue.empty():
                sleep(0)
            else:
                print('went into else statement because p_queue wasnt empty') 
                s_id, params = p_queue.get()
                r_queue.put((s_id, r_gen.rollout(params)))
                print("rqueue is not empty anymore")


################################################################################
#                Define queues and start workers                               #
################################################################################
p_queue = Queue()
r_queue = Queue()
e_queue = Queue()

for p_index in range(num_workers):
    print('defined stuff') 
    Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index)).start()
    print('after process started')


################################################################################
#                           Evaluation                                         #
################################################################################
def evaluate(solutions, results, rollouts=100):
    """ Give current controller evaluation.

    Evaluation is minus the cumulated \ averaged over rollout runs.

    :args solutions: CMA set of solutions
    :args results: corresponding results
    :args rollouts: number of rollouts

    :returns: minus averaged cumulated reward
    """
    index_min = np.argmin(results)
    best_guess = solutions[index_min]
    restimates = []

    for s_id in range(rollouts):
        p_queue.put((s_id, best_guess))

    print("Evaluating...")
    for _ in tqdm(range(rollouts)):
        while r_queue.empty():
            sleep(0)
        restimates.append(r_queue.get()[1])

    return best_guess, np.mean(restimates), np.std(restimates)

################################################################################
#                           Launch CMA                                         #
################################################################################
controller = Controller(LSIZE, RSIZE, ASIZE)  # dummy instance

train_reward_list = []
test_reward_list = [] 

# define current best and load parameters
cur_best = None
ctrl_file = join(ctrl_dir, 'best.tar')
print("Attempting to load previous best...")
if exists(ctrl_file):
    state = torch.load(ctrl_file, map_location={'cuda:0': 'cpu'})
    cur_best = - state['reward']
    controller.load_state_dict(state['state_dict'])
    print("Previous best was {}...".format(-cur_best))

parameters = controller.parameters()
es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1,
                              {'popsize': pop_size})

epoch = 0
log_step = 3

while not es.stop():
    if cur_best is not None and - cur_best > args.target_return:
        print("Already better than target, breaking...")
        break

    r_list = [0] * pop_size  # result list
    solutions = es.ask()

    # push parameters to queue
    for s_id, s in enumerate(solutions):
        for _ in range(n_samples):
            p_queue.put((s_id, s))
            
    print('parameters pushed into p_queue') 

    # retrieve results
    if args.display:
        pbar = tqdm(total=pop_size * n_samples)
    for _ in range(pop_size * n_samples):
        while r_queue.empty():
            sleep(0)
        r_s_id, r = r_queue.get()
        r_list[r_s_id] += r / n_samples
        if args.display:
            pbar.update(1)
    if args.display:
        pbar.close()

    es.tell(solutions, r_list)
    es.disp()
    
    train_reward_list.append(r_list)
    pickle.dump((train_reward_list), open('train_controller_on_vae_vanilla_train.p', 'wb'))

    # evaluation and saving
    if epoch % log_step == log_step - 1:
        best_params, best, std_best = evaluate(solutions, r_list)
        test_reward_list.append(best)
        #pickle.dump((reward_list), open('train_controller_on_vaernn.p', 'wb'))
        #pickle.dump((reward_list), open('train_controller_on_vae_reward_eval_pop64.p', 'wb'))
        pickle.dump((test_reward_list), open('train_controller_on_vae_vanilla_test.p', 'wb'))
        print("Current evaluation: {}".format(best))
        if not cur_best or cur_best > best:
            cur_best = best
            print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
            load_parameters(best_params, controller)
            torch.save(
                {'epoch': epoch,
                 'reward': - cur_best,
                 'state_dict': controller.state_dict()},
                join(ctrl_dir, 'best.tar'))
        if - best > args.target_return:
            print("Terminating controller training with value {}...".format(best))
            break

    epoch += 1
    if epoch == 80:
        break 

es.result_pretty()
e_queue.put('EOP')
