from ast import Assign
from urllib import response
import numpy as np
import numpyro
import pandas as pd
import numpy as np
import os
import argparse
import pickle
import time
import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.special import logit
from numpyro.distributions import Gamma, Beta, Normal, Categorical
from numpyro.distributions.transforms import StickBreakingTransform
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
from scipy.special import logit, expit
from scipy import stats
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


def get_consensus(test_excel):
    # test_excel = test_excel.drop(test_excel.columns[[0]], axis=1)
    stacked = test_excel.stack().reset_index()
    stacked.columns = ['Row', 'Column', 'Value']
    result = stacked.apply(lambda x: [x['Row'], test_excel.columns.get_loc(x['Column']), x['Value']], axis=1).tolist()

    # Convert list to numpy array
    response_matrix = np.array(result, dtype=float)
    response_matrix[response_matrix[:, 2] == 10, 2] = 9.99
    response_matrix[response_matrix[:, 2] == 0, 2] = 0.01

    # Model
    def model():
        no_cultures = 3
        alpha = numpyro.sample('alpha', Gamma(1, 1))
        with numpyro.plate('weights', no_cultures - 1):
            v = numpyro.sample('v', Beta(1, alpha))
            
        with numpyro.plate('ind', participants):
            sb_trans = StickBreakingTransform()
            culture_id = numpyro.sample("culture_id", Categorical(sb_trans(v)))
            
        with numpyro.plate("data_loop", participants):
            culture_assignment = culture_id[features[:, 0]]
            
        with numpyro.plate('parameters', no_cultures):
            competence_mean = numpyro.sample('competence_mean', Normal(0, 4))
            competence_precision = numpyro.sample('competence_precision', Gamma(0.01, 0.01))
            scale_prior_precision = numpyro.sample("scale_prior_precision", Gamma(0.01, 0.01))
            bias_prior_variance = numpyro.sample("bias_prior_variance", Gamma(0.01, 0.01))
            consensus_mean = numpyro.sample('consensus_mean', Normal(0, 4))
            consensus_precision = numpyro.sample('consensus_precision', Gamma(0.01, 0.01))
            itemDiff_precision = numpyro.sample('itemDiff_precision', Gamma(0.01, 0.01))
            with numpyro.plate('culture_ind', participants):
                LogCompetence = numpyro.sample('LogCompetence', Normal(competence_mean, 1/competence_precision))
                bias = numpyro.sample('bias', Normal(0, 1/bias_prior_variance))
                LogScale = numpyro.sample('LogScale', Normal(0, 1/scale_prior_precision))
            with numpyro.plate('stimplate', stimuli):
                consensus = numpyro.sample('consensus', Normal(consensus_mean, 1/consensus_precision))
                itemDiff = numpyro.sample('itemDiff', Normal(0, 1/itemDiff_precision))
        competence = jnp.exp(LogCompetence[features[:, 0], culture_assignment])    
        scale = jnp.exp(LogScale[features[:, 0], culture_assignment])
        item_difficulty = jnp.exp(itemDiff[features[:, 1], culture_assignment])
        ratingMu = vmap(lambda scale, cons, bias: scale * cons + bias)(scale, consensus[features[:, 1], culture_assignment], bias[features[:, 0], culture_assignment])
        ratingVariance = (scale * competence * item_difficulty)**2
        
        
        numpyro.sample("rating", Normal(ratingMu, ratingVariance), obs=rates)

    # Data features
    features, rates = response_matrix[:, :].astype(int), logit(response_matrix[:, 2].astype(float)/10)
    participants = len(jnp.unique(response_matrix[:, 0]))
    stimuli = len(jnp.unique(response_matrix[:, 1]))
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))

    # Model specifications
    kernel = DiscreteHMCGibbs(NUTS(model, target_accept_prob = 0.80, max_tree_depth=6))
    mcmc = MCMC(
        kernel,
        num_warmup= 500,
        num_samples= 500,
        num_chains= 1,
        chain_method= "sequential",
        progress_bar=True
    )
    # Run the model
    mcmc.run(rng_key)

    # Get the results
    diagnos = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True))
    with open('intervention-test.pickle', 'wb') as handle:
        pickle.dump(diagnos, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Save cultural assignment
    samples = mcmc.get_samples()
    np.save('cultural_assignments.npy', samples['culture_id'])
    culture_assignment = np.load('cultural_assignments.npy')
    mode_culture_assignment = culture_assignment
    mode_cultures = stats.mode(mode_culture_assignment)
    assigment_mode_culture = mode_cultures[0]
    # assigment_mode_culture includes cultural assignments that matches with participant id index from 0....N. 
    print(assigment_mode_culture)

    with open('intervention-test.pickle', 'rb') as handle:
        posterior_inference = pickle.load(handle)
        
    consensus = expit(posterior_inference['consensus']['mean'])
    competence = posterior_inference['LogCompetence']['mean']
    question_difficulty = posterior_inference['itemDiff']['mean']
    
    # Find the most competent participant in each culture
    cultures = np.unique(assigment_mode_culture)
    comp_dict = {culture: [] for culture in cultures}

    for participant in range(participants):
        participant_culture = assigment_mode_culture[participant]
        participant_competence = competence[participant][participant_culture]
        comp_dict[participant_culture].append((participant, participant_competence))
    
    highest_competent = {culture: None for culture in cultures}


    for culture_id, participants in comp_dict.items():
        if participants:  # Check if there are any participants in this culture
            highest_competent[culture_id] = max(participants, key=lambda x: x[1])[0]  # Get the participant ID with the highest competence
    print('the highest competent individuals: ')

    print(highest_competent) # map culture id to participant id with highest competence
    
    consensus_dict = {culture: [] for culture in cultures} # map culture id to consensus values

    # cultures is a list of unique culture ids
    
    for c in cultures:
        consensus_dict[c].append(expit(consensus[:,c]))
    
    data = np.array([consensus_dict[c] for c in cultures])
    variances = np.var(data, axis=0)
    
    print('variances: ', variances)
    print('consensus: ', consensus_dict)

    max_variance_index = np.argmax(variances)
    max_variance_value = variances[0][max_variance_index]
    print('the most variant question: ' + str(max_variance_index) + ' with variance: ' + str(max_variance_value))

    light_colors = [
        '#FFA07A',  # Light Salmon
        '#90EE90',  # Light Green
        '#87CEFA',  # Light Sky Blue
        '#FFB6C1',  # Light Pink
        '#FAFAD2',  # Light Goldenrod Yellow
        '#E6E6FA',  # Lavender
        '#E0FFFF',  # Light Cyan
        '#D3D3D3',  # Light Gray
        '#20B2AA',  # Light Sea Green
        '#F08080'   # Light Coral
    ]
    
    colorMap = {} # a map of culture id to a color string
    cultureMap = {} # a map of culture id to a list of consensus values
    for i in range(len(cultures)):
        colorMap[str(cultures[i].item())] = light_colors[i]

    for culture in cultures:
        listy = [round(c*10, 0) for c in consensus[:,culture].tolist()]
        cultureMap[str(culture.item())] = listy

    
    assigment_mode_culture = [str(s) for s in assigment_mode_culture] # map participant id to culture id
    return [cultureMap, colorMap, assigment_mode_culture, float(max_variance_index), list(highest_competent.values())]
# test_excel = pd.read_excel('test_excel.xlsx')
# get_consensus(test_excel)
