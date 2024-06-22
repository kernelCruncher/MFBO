import pandas as pd
import numpy as np
import torch

from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP, SingleTaskGP
from botorch.posteriors.gpytorch import scalarize_posterior
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy
from botorch.acquisition import PosteriorMean 
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim.optimize import optimize_acqf
import numpy as np
from scipy.spatial.distance import cdist
from botorch import fit_gpytorch_mll
torch.set_printoptions(precision=12, sci_mode=False)
import copy
import math
import matplotlib.pyplot as plt
import random
import time
import pickle
import os
import warnings

warnings.filterwarnings("ignore")


# %%
def main(sample_space_size):
    
    file = setUpSampleSpace(sample_space_size)
    N_INIT = 5
    ALLOCATED_BUDGET = 100

    fileName = file

    train_x_full, train_obj, domain, index_store, fidelity_history = setUpInitialData(fileName, N_INIT)
    train_x_full_sf, train_obj_sf, domain_sf, index_store_sf, fidelity_history_sf = convertMFDatatoSFData(domain, index_store)

    train_x_full_tvr, train_obj_tvr, cumulative_cost_tvr, index_store_tvr = run_entire_cycle(
        train_x_full, 
        train_obj, 
        domain, 
        fidelity_history,
        index_store,
        runTVR,
        allocated_budget=ALLOCATED_BUDGET
        )

    train_x_full_mes, train_obj_mes, cumulative_cost_mes, index_store_mes = run_entire_cycle(
        train_x_full, 
        train_obj, 
        domain, 
        fidelity_history,
        index_store,
        runMes,
        allocated_budget=ALLOCATED_BUDGET
        )

    train_x_full_kg, train_obj_kg, cumulative_cost_kg, index_store_kg = run_entire_cycle(
        train_x_full, 
        train_obj, 
        domain, 
        fidelity_history,
        index_store,
        runKG,
        allocated_budget=ALLOCATED_BUDGET
        )

    train_x_full_rand, train_obj_rand, cumulative_cost_rand, index_store_rand= run_entire_cycle_random(ALLOCATED_BUDGET, domain, train_x_full_sf)

    train_x_full_sfmes, train_obj_sfmes, cumulative_cost_sfmes, index_store_sfmes= run_entire_cycle(
        train_x_full_sf, 
        train_obj_sf, 
        domain_sf,
        fidelity_history_sf,
        index_store_sf,
        runMes,
        sf=True,
        allocated_budget=ALLOCATED_BUDGET)

    train_x_full_ei, train_obj_ei, cumulative_cost_ei, index_store_ei= run_entire_cycle(
        train_x_full_sf, 
        train_obj_sf, 
        domain_sf,
        fidelity_history_sf,
        index_store_sf,
        runEI,
        sf=True,
        allocated_budget=ALLOCATED_BUDGET)
    
    modelDict = {
    "MF-TVR": (train_x_full_tvr, train_obj_tvr, cumulative_cost_tvr),
    "MF-KG": (train_x_full_kg, train_obj_kg, cumulative_cost_kg),
    "MF-MES": (train_x_full_mes, train_obj_mes, cumulative_cost_mes),
    "SF-MES": (train_x_full_sfmes, train_obj_sfmes, cumulative_cost_sfmes),
    "SF-EI" : (train_x_full_ei, train_obj_ei, cumulative_cost_ei),
    "Random": (train_x_full_rand, train_obj_rand, cumulative_cost_rand), 
             }
    
    save_dictionary(modelDict, batch=False)    

def covSEard(hyp, x, z):
    """
    ARD covariance:
        x is of dimension n X D
        y is of dimension m X D
    """
    hyp = np.exp(hyp)

    D = x.shape[1]
    X = (1 / hyp[:D]) * x

    Z = (1 / hyp[:D]) * z
    K = cdist(X, Z)

    K = hyp[D] ** 2 * np.exp(-K ** 2 / 2)

    return K

def rkhs_synth(x):
    """
    RKHS Function
        Description: Synthetic heteroscedastic function generated from 2 Squared Exponential kernels
                     for Bayesian Optimization method evaluation tasks
        Evaluated: x \in [0,1]
        Global Maximum: x=0.89235, f(x)=5.73839
        Authors: Ziyu Wang, John Assael and Nando de Freitas
    """

    x = np.atleast_2d(x)
    hyp_1 = np.log(np.array([0.1, 1]))
    hyp_2 = np.log(np.array([0.01, 1]))

    support_1 = [0.1, 0.15, 0.08, 0.3, 0.4]
    support_2 = [0.8, 0.85, 0.9, 0.95, 0.92, 0.74, 0.91, 0.89, 0.79, 0.88, 0.86, 0.96, 0.99, 0.82]
    vals_1 = [4, -1, 2., -2., 1.]
    vals_2 = [3, 4, 2, 1, -1, 2, 2, 3, 3, 2., -1., -2., 4., -3.]

    f = sum([vals_2[i] * covSEard(hyp_2, np.atleast_2d(np.array(s)), x) for i, s in enumerate(support_2)])
    f += sum([vals_1[i] * covSEard(hyp_1, np.atleast_2d(np.array(s)), x) for i, s in enumerate(support_1)])

    return float(f)

def setUpSampleSpace(spaceSize=200):
    Xpr = np.linspace(0,1,spaceSize)
    
    domain = []
    for x in Xpr:
       domain.append( [x, 1.0, rkhs_synth(x)])
       domain.append( [x, 0.5, rkhs_synth(x) + random.gauss(0, 1)])
    
    domain = np.array(domain)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs('SampleSpaces', exist_ok=True)
    fileName = 'SampleSpaces/'+timestr + '.csv'
    np.savetxt(fileName, domain, delimiter=',')

    return fileName

def setUpInitialData(sampleSpaceName, initialSize=10, predefined_indices = None, sf=False, file=True):
      # The file argument is telling us whether we expect the sampleSpaceName to be a file or the actual domain is already in memory.
      # The predefined_indices argument us used in the batch case acorss multiple search-algorithms where we want 
      #  each element in the batch to have the same intitial set up so that we can compare the averages fairly.
      sampleSpace = np.loadtxt(sampleSpaceName, delimiter=',') if file else sampleSpaceName
      if predefined_indices is None:  
            sampleSpace_hf = sampleSpace[np.where(sampleSpace[:, 1]==1)]
            size = len(sampleSpace_hf)
            index_store = random.sample(range(size), initialSize)
            #This gets the high fidelity and low fidelity points in pairs if we're doing MF.
            sampleSpace, index_store = (sampleSpace_hf, index_store) if sf else (sampleSpace, [2 * x  for x in  index_store] + [1 + 2 * x for x in index_store])
            fidelity_history = sampleSpace[index_store, 1]
            train_X = sampleSpace[index_store, :-1]
            train_obj = sampleSpace[index_store, -1:]
            return torch.tensor(train_X), torch.tensor(train_obj), sampleSpace, index_store, fidelity_history.flatten().tolist()
      else:
            fidelity_history = sampleSpace[predefined_indices, 1]
            train_X = sampleSpace[predefined_indices, :-1]
            train_obj = sampleSpace[predefined_indices, -1:]
            return torch.tensor(train_X), torch.tensor(train_obj), sampleSpace, predefined_indices, fidelity_history.flatten().tolist()

# Required when we want to ensure that the sf has the same hf points in its intitial sampel as the mf case.
def convertMFDatatoSFData(sampleSpace, indexStore):
      sampleSpace_hf = sampleSpace[np.where(sampleSpace[:, 1]==1)]
      index_store = [x // 2 for x in indexStore if x % 2 == 0]
      return torch.tensor(sampleSpace_hf[index_store, : -1]), torch.tensor(sampleSpace_hf[index_store, -1:]), sampleSpace_hf, index_store, sampleSpace[index_store, 1].flatten().tolist()
    
def save_dictionary(dictionary, batch=False, root='SearchDictionaries'):
      os.makedirs(root, exist_ok=True)
      timestr = time.strftime("%Y%m%d-%H%M%S")
      fileName = root + '/' + 'Batch_' + timestr if batch else root + '/' + timestr
      with open(fileName, 'wb') as handle:
         pickle.dump(dictionary, handle)
      return fileName

def load_dictionary(file):
    with open(file, 'rb') as inp:
      output = pickle.load(inp)
      return output
    
def create_correlation_dict(no_points, corr_parameters):
    range_100 = np.linspace(0, 1, no_points)
    high_fid = np.array([ rkhs_synth(x) for x in range_100])
    corr_dict = {'base': range_100, '1': high_fid}
    gaussian_noise = np.array([random.gauss(0, 50) for x in range(0, no_points) ])
    for n in corr_parameters:
        low_fid = np.add(high_fid, 1/n * gaussian_noise)
        correlation = np.corrcoef(high_fid, low_fid)[0,1]
        corr_dict[str(correlation)] = low_fid
    return corr_dict

def runMes(model, Xrpr, previous_evaluations=None, train_x_past=None):
    Xrpr = torch.tensor(Xrpr)
    bounds = torch.tensor([[0.0] * Xrpr.shape[1], [1.0] * Xrpr.shape[1]])
    candidate_set = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(10000, 1)
    target_fidelities = {1: 1.0}
            
    cost_model = AffineFidelityCostModel(fidelity_weights={1: 1.0}, fixed_cost=1.0)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    acquisition = qMultiFidelityMaxValueEntropy(
            model=model,
            cost_aware_utility=cost_aware_utility,
            project=lambda x: project_to_target_fidelity(X=x, target_fidelities=target_fidelities),
            candidate_set=candidate_set,
        )
    acquisitionScores =  acquisition.forward(Xrpr.reshape(-1,1, Xrpr.shape[1]))
    return acquisitionScores

def runKG(model, Xrpr, previous_evaluations=None, train_x_past=None):
    Xrpr = torch.tensor(Xrpr)
    bounds = torch.tensor([[0.0] * Xrpr.shape[1], [1.0] * Xrpr.shape[1]])
    target_fidelities = {1: 1.0}
            
    cost_model = AffineFidelityCostModel(fidelity_weights={1: 1.0}, fixed_cost=1.0)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=Xrpr.shape[1],
        columns=[Xrpr.shape[1]-1],
        values=[1],
    )                
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:,:-1],
        q=1,
        num_restarts= 2,
        raw_samples=4
    )
    acquisition = qMultiFidelityKnowledgeGradient(
            model=model,
            cost_aware_utility=cost_aware_utility,
            project=lambda x: project_to_target_fidelity(X=x, target_fidelities=target_fidelities),
            current_value=current_value,
            num_fantasies= 5
        )
    acquisitionScores =  acquisition.evaluate(Xrpr.reshape(-1,1, Xrpr.shape[1]), bounds=bounds).detach()
    return acquisitionScores

def runEI(model, Xrpr, previous_evaluations, train_x_past=None):
    Xrpr = torch.tensor(Xrpr)
    acquisition = ExpectedImprovement(
            model=model,
            best_f= max(previous_evaluations)
        )
    
    acquisitionScores =  acquisition.forward(Xrpr.reshape(-1,1, Xrpr.shape[1]) ).detach()
    return acquisitionScores

def runTVR(model, Xrpr, previous_evaluations=None, train_x_past=None):
    Xrpr_hf = Xrpr[np.where(Xrpr[:, 1]==1)]
    indices = np.where(train_x_past[:, 1] == 1)

    acquisition_scores = runEI(model, Xrpr_hf, previous_evaluations)
    max_hf_ind = acquisition_scores.argmax()

    index_in_xrpr = Xrpr.tolist().index(Xrpr_hf[max_hf_ind].tolist())
    Xrpr = torch.tensor(Xrpr)

    posterior = model.posterior(Xrpr)

    pcov = posterior.distribution.covariance_matrix
    p_var = posterior.variance
    hf_max_cov = pcov[index_in_xrpr]
    hf_max_var = hf_max_cov[index_in_xrpr]
    cost = Xrpr[:, 1]
    
    return hf_max_cov ** 2 / (p_var.reshape(-1) * hf_max_var * cost)   

def optimiseAcquisitionFunction(sortedAcqusitionScores, domain, trainingData, index_store):
    # X_detached = trainingData.detach().numpy()
    # def checkFunction(candidate, set):
    #     for x in set:
    #         if np.array_equal(candidate[:-1], x):
    #             return True
    #     return False
    def checkIndexNotAlreadyEvaluated(candidate, set):
        return candidate in set
    
    for i in range(domain.shape[0]):
        if not checkIndexNotAlreadyEvaluated(sortedAcqusitionScores[i].item(), index_store):
            index_store.append(sortedAcqusitionScores[i].item())
            return domain[sortedAcqusitionScores[i], 0], domain[sortedAcqusitionScores[i], 1], domain[sortedAcqusitionScores[i], 2]
            # , sortedAcqusitionScores[i]    

def run_entire_cycle(train_x_full, 
                     train_obj, 
                     domain, 
                     fidelity_history, 
                     index_store, 
                     func,
                     sf=False, 
                     no_of_iterations=100000, 
                     allocated_budget=100000
                     ):
    train_x_full = copy.deepcopy(train_x_full)
    train_obj = copy.deepcopy(train_obj)
    fidelity_history = copy.deepcopy(fidelity_history)
    index_store = copy.deepcopy(index_store)
    
    domain_X_only = domain[:, 0:-1]
    budget_sum = sum(fidelity_history)
    iteration_counter = 0
    while budget_sum  <= allocated_budget - 1 and iteration_counter < no_of_iterations: 
        # The - 1 important in the budget (as well as the equal) as the check happens at the start and we only really care about high-fidelity points.
        # Consider a budget of 40, and when we hit the sum at 39. We would want the subsequent step to be the last
        # as at most we can add 1. If we instead only add 0.5, you could argue that stopping at 39.5 is premature
        # and we could go another step since it's possible to get another low-fidelity point, but this does not interest us.
        # It's the high-fidelity points we care about and then that would exceed the budget.
        model = SingleTaskGP(train_x_full, train_obj) if sf else SingleTaskMultiFidelityGP(train_x_full, train_obj, data_fidelity=1)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)  
        acquisitionScores = func(model=model, Xrpr=domain_X_only, previous_evaluations=train_obj, train_x_past=train_x_full )
        sorted_acqusition_scores = acquisitionScores.argsort(descending=True)
        top_candidate, fidelity, evaluation = optimiseAcquisitionFunction(sorted_acqusition_scores, domain, train_x_full, index_store)
        fidelity_history.append(fidelity)
        train_x_full = torch.cat([train_x_full, torch.tensor([top_candidate, fidelity]).unsqueeze(0)])
        train_obj = torch.cat([train_obj, torch.tensor([evaluation]).unsqueeze(-1)])
        iteration_counter+=1
        budget_sum += fidelity
    cumulative_cost = [fidelity_history[0]]
    for i in range(len(fidelity_history) - 1):
        cumulative_cost.append(cumulative_cost[-1] + fidelity_history[i+1])
    return train_x_full, train_obj, cumulative_cost, index_store

def run_entire_cycle_random(no_of_iterations, domain, x_train=None):
    if x_train is None:                   
        #Here we will only consider high-fidelity points since we are just randomly choosing points and ignore the intitial sample.
        #Since high-fidelity points have a fidelity of 1 this means the allocated_budget is the same as the number of iterations.
        high_fidelity_points = domain[np.where(domain[:, 1] == 1.0)]
        
        number_of_hf_points = len(high_fidelity_points)
        index_store = random.sample(range(number_of_hf_points), no_of_iterations)
        train_X_full = high_fidelity_points[index_store][:, :-1]
        train_obj = high_fidelity_points[index_store][:, -1]
        cumulative_cost = list(range(1, no_of_iterations + 1))
        return torch.tensor(train_X_full), torch.tensor(train_obj).unsqueeze(-1), cumulative_cost, [2 * x  for x in index_store]
    else:
        # This is for when we want to do a random search, but keep the initial points already computed in the initial sampling used by other techniques.
        # The benefit of this is when we wish to compare precisely how the paths diverged for different techniques.
        no_of_iterations_left =  no_of_iterations - len(x_train)

        high_fidelity_points =  np.array([x for x in domain if ((x[:-1].tolist() not in x_train.tolist()) and (x[1] == 1.0)) ])        
        
        number_of_hf_points = len(high_fidelity_points)
        index_store = random.sample(range(number_of_hf_points), no_of_iterations_left)
        train_X_full = torch.cat([x_train, torch.tensor(high_fidelity_points[index_store][:, :-1])])
        index_store = [domain[:, :-1].tolist().index(x) for x in train_X_full.tolist()]
        train_obj = domain[index_store][:, -1]
        cumulative_cost = list(range(1, no_of_iterations + 1))
        return train_X_full, torch.tensor(train_obj).unsqueeze(-1), cumulative_cost, index_store

if __name__ == "__main__":
    from argparse import ArgumentParser
  
    parser = ArgumentParser()
    parser.add_argument("--sample_space_size", type=int, default=100)
    args = parser.parse_args()
    main(
        args.sample_space_size,
    )
