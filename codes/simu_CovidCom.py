from SGShift import *
import pandas as pd
import numpy as np
import os
df = pd.read_csv('/net/dali/home/mscbio/rul98/PheOpt/compare/whyshift/datasets/new_CovidCom.csv', index_col= 0)
df['ethnicity_Hispanic'] = np.where(df['ethnicity'] == 'Hispanic or Latino', 1, 0)
df['ethnicity_NotHispanic'] = np.where(df['ethnicity'] == 'Not Hispanic or Latino ', 1, 0)
df['age_group'] = pd.cut(df['age'], bins=[0, 17, 44, 64, max(df['age'])], labels=['0-17', '18-44', '45-64', '65-'], right=False)
df['race'] = df['race'].astype('category')
df['gender'] = df['gender'].astype('category')

df['race_White'] = (df['race'] == 'White').astype(int)
df['race_Black_or_African_American'] = (df['race'] == 'Black or African American').astype(int)
df['race_More_than_one_population'] = (df['race'] == 'More than one population').astype(int)
df['race_Asian'] = (df['race'] == 'Asian').astype(int)
df['race_American_Indian_or_Alaska_Native'] = (df['race'] == 'American Indian or Alaska Native').astype(int)

df['gender_Female'] = (df['gender'] == 'Female').astype(int)
df['gender_Male'] = (df['gender'] == 'Male').astype(int)
df['0-17'] = (df['age_group'] == '0-17').astype(int)
df['18-44'] = (df['age_group'] == '18-44').astype(int)
df['45-64'] = (df['age_group'] == '45-64').astype(int)
df['65-'] = (df['age_group'] == '65-').astype(int)

df['observation_datetime'] = pd.to_datetime(df['observation_datetime'], errors='coerce')
df['month'] = df['observation_datetime'].dt.to_period('M')

feature_names = ['EHR', 'Claims', 'Primary', 'Secondary', 'E03',
    'E11', 'E78', 'I10', 'I25', 'I50', 'J44', 'J45', 'J96', 'N18', 'R06',
    'R07', 'R09', 'ethnicity_Hispanic',
    'ethnicity_NotHispanic', 'race_White',
    'race_Black_or_African_American', 'race_More_than_one_population',
    'race_Asian', 'race_American_Indian_or_Alaska_Native', 'gender_Female',
    'gender_Male', '0-17', '18-44', '45-64', '65-']

cutoff_before = pd.Period('2022-01')
cutoff_after = pd.Period('2022-02')

source_df = df[(df['month'] >= pd.Period('2021-01')) & (df['month'] < cutoff_before)]
target_df = df[(df['month'] >= cutoff_before) & (df['month'] < cutoff_after)]
val_df = df[(df['month'] >= cutoff_after) & (df['month'] < pd.Period('2023-02'))]

feature_names = np.array(feature_names)
X_S = source_df[feature_names].values
y_S = source_df['inpatient'].values
X_T = target_df[feature_names].values
y_T = target_df['inpatient'].values
X_val = val_df[feature_names].values
y_val = val_df['inpatient'].values

args = parse_arguments()
X_S, y_S = matchsample(X_S, y_S, random_seed=args.random_state)
X_T, y_T = downsample(X_T, y_T, n_samples = args.n_test, random_seed=args.random_state)
X_val, y_val = matchsample(X_val, y_val, random_seed=args.random_state)
directory = args.directory + str(args.n_test)
if not os.path.exists(directory):
    os.makedirs(directory)
lambdas = np.geomspace(0.0001, 1, num = 13)
sel_thrs = np.linspace(0.05, 1.0, 20)
results_naive = []
results_absorption = []
ll1 = np.array([0.0003, 0.001, 0.003])
results_naive_knock = {l:[] for l in ll1}
ll2 = np.array([0.0001, 0.0003])
results_knock_absorption = {l:[] for l in ll2}
ll3 = np.geomspace(0.00001, 100, num = 29)
results_diff = []
results_shap = []
results_whyshift = []
all_sel_whyshift = []
results_source = []
results_target = []
rng = check_random_state(args.random_state)
n_S, p = X_S.shape
n_T = X_T.shape[0]
n_val = X_val.shape[0]
y_T_val_freq = (y_T.sum() + y_val.sum()) / (len(y_T) + len(y_val))
shifted_feature = np.array(['Claims', 'Primary', 'E11', 'I10', 'I25', 'J96', 'R06', 'R09', 'gender_Female', '45-64'])
shift_subset_y = np.where(np.isin(feature_names, shifted_feature))[0]
for b in range(args.B):
    simulated = simulate_data(
        task=args.task,
        model_type=args.model,
        X_S = X_S, 
        X_T_val = np.concatenate([X_T, X_val]),
        y_S_tmp = y_S, 
        y_T_val_freq = y_T_val_freq,
        n_S=n_S,
        n_T=n_T,
        n_val=n_val,
        p=p,
        p_informative=None,
        shift_features_X=None,
        shift_magnitude_X=0.,
        shift_subset_y=shift_subset_y,
        shift_magnitude_y=args.shift_magnitude_y,
        linear_shift_y = True,
        random_state=args.random_state+b
    )
    X_S = simulated['source']['X']
    y_S = simulated['source']['y']
    X_T = simulated['target']['X'] 
    y_T = simulated['target']['y']
    X_val = simulated['validation']['X']
    y_val = simulated['validation']['y']
    
    solver = fit_model(args.task, args.solver, args.random_state)
    source_only = deepcopy(solver)
    source_only.fit(X_S, y_S)
    target_only = deepcopy(solver)
    target_only.fit(X_T, y_T)
    
    p_T = model2p(source_only, X_T)
    p_source_only = model2p(source_only, X_val)
    p_target_only = model2p(target_only, X_val)
    f_x_T = np.log(p_T / (1 - p_T))
    f_x_val = np.log(p_source_only / (1 - p_source_only))
    eval_results_source = evaluation(y_val, p_source_only, task = args.task)
    results_source.append(eval_results_source)
    eval_results_target = evaluation(y_val, p_target_only, task = args.task)
    results_target.append(eval_results_target)
    pd.DataFrame(results_source).to_csv(f'{directory}/source_{args.model}_{args.solver}.csv', index=False)
    pd.DataFrame(results_target).to_csv(f'{directory}/target_{args.model}_{args.solver}.csv', index=False)
    
    for lambda_reg in lambdas:
        delta_estimated = estimate_delta(X_T, y_T, lambda_reg, source_only, task = args.task)
        selected_naive = np.where(np.abs(delta_estimated) > 1e-4)[0]
        x_delta = X_val @ delta_estimated
        shifted_log_odds = f_x_val + x_delta
        p_val_naive = 1 / (1 + np.exp(-shifted_log_odds))
        eval_results_naive = evaluation(y_val, p_val_naive, shift_subset_y, selected_naive, task = args.task)
        eval_results_naive['lambda_reg'] = lambda_reg
        results_naive.append(eval_results_naive)
        df = pd.DataFrame(results_naive)
        df.to_csv(f'{directory}/naive_{args.model}_{args.solver}.csv', index=False)

    pi = derandom_knock(solver, X_S, y_S, X_T, y_T, 20, ll1, task = args.task)
    for l_ in ll1:
        for sel_thr in sel_thrs:
            selected_naive_knock = np.where(pi[l_] >= sel_thr)[0]
            if len(selected_naive_knock) > 0: 
                delta_estimated = estimate_delta(X_T[:, selected_naive_knock], y_T, l_, f_x = f_x_T, task = args.task)
                x_delta = X_val[:, selected_naive_knock] @ np.array(delta_estimated)
                shifted_log_odds = f_x_val + x_delta
            else: 
                shifted_log_odds = f_x_val
            p_val_naive_knock = 1 / (1 + np.exp(-shifted_log_odds))
            eval_results_naive_knock = evaluation(y_val, p_val_naive_knock, shift_subset_y, selected_naive_knock, task = args.task)
            eval_results_naive_knock['sel_thr'] = sel_thr
            results_naive_knock[l_].append(eval_results_naive_knock)
            df_results_naive_knock = pd.DataFrame(results_naive_knock[l_])
            df_results_naive_knock.to_csv(f'{directory}/naive_knock_{args.model}_{args.solver}_{l_}.csv', index=False)
    
    for lambda_reg in lambdas:
        lambda_reg = lambda_reg / 10
        lambda_beta = 0.9 * lambda_reg
        beta_est, delta_est = estimate_beta_delta(X_S, y_S, X_T, y_T, lambda_beta, lambda_reg, source_only, task = args.task)
        selected_miss = np.where(np.abs(delta_est) > 1e-4)[0]
        x_beta = X_val @ beta_est
        x_delta = X_val @ delta_est
        shifted_log_odds = f_x_val + x_beta + x_delta
        p_val_miss = 1 / (1 + np.exp(-shifted_log_odds))
        eval_results_absorption = evaluation(y_val, p_val_miss, shift_subset_y, selected_miss, task = args.task)
        eval_results_absorption['lambda_reg'] = lambda_reg
        results_absorption.append(eval_results_absorption)
        df = pd.DataFrame(results_absorption)
        df.to_csv(f'{directory}/absorption_{args.model}_{args.solver}.csv', index=False)
    
    pi = derandom_knock_misspecified(solver, X_S, y_S, X_T, y_T, 20, ll2, task = args.task)
    for l_ in ll2:
        beta_estimated, delta_estimated = estimate_beta_delta(X_S, y_S, X_T, y_T, 0.9*l_, l_, source_only, task = args.task)
        for sel_thr in sel_thrs:
            selected_knock_absorption = np.where(pi[l_] >= sel_thr)[0]
            if len(selected_knock_absorption) > 0: 
                delta_estimated = estimate_delta(X_T[:, selected_knock_absorption], y_T, l_, f_x = (f_x_T + X_T @ np.array(beta_estimated)), task = args.task)
                x_delta = X_val[:, selected_knock_absorption] @ np.array(delta_estimated)
                shifted_log_odds = f_x_val + X_val @ np.array(beta_estimated) + x_delta
            else:
                shifted_log_odds = f_x_val
            p_val_knock_absorption = 1 / (1 + np.exp(-shifted_log_odds))
            eval_results_knock_absorption = evaluation(y_val, p_val_knock_absorption, shift_subset_y, selected_knock_absorption, task = args.task)
            eval_results_knock_absorption['sel_thr'] = sel_thr
            results_knock_absorption[l_].append(eval_results_knock_absorption)
            df_results_knock_absorption = pd.DataFrame(results_knock_absorption[l_])
            df_results_knock_absorption.to_csv(f'{directory}/absorption_knock_{args.model}_{args.solver}_{l_}.csv', index=False)
    
    X_S_train, X_S_test, y_S_train, y_S_test = train_test_split(X_S, y_S, test_size=0.5, random_state=args.random_state)
    X_T_train, X_T_test, y_T_train, y_T_test = train_test_split(X_T, y_T, test_size=0.5, random_state=args.random_state)
    source_only.fit(X_S_train, y_S_train)
    target_only.fit(X_T_train, y_T_train)
    
    X_stack = np.vstack([X_S_test, X_T_test])
    proba_source_only = model2p(source_only, X_stack)
    proba_target_only = model2p(target_only, X_stack)
    f_source_only = np.log(proba_source_only / (1 - proba_source_only))
    f_target_only = np.log(proba_target_only / (1 - proba_target_only))
    f_diff = f_target_only - f_source_only
    
    for l_ in ll3:
        diff_model = Lasso(alpha=l_, max_iter=10000, random_state=args.random_state).fit(X_stack, f_diff)
        diff_estimated = diff_model.coef_
        selected_diff = np.where(abs(diff_estimated) > 1e-4)[0]
        shifted_log_odds = f_x_val + X_val @ np.array(diff_estimated)
        p_val_diff = 1 / (1 + np.exp(-shifted_log_odds))
        eval_results_diff = evaluation(y_val, p_val_diff, shift_subset_y, selected_diff, task = args.task)
        eval_results_diff['lambda_reg'] = l_
        results_diff.append(eval_results_diff)
        df_results_diff = pd.DataFrame(results_diff)
        df_results_diff.to_csv(f'{directory}/diff_{args.model}_{args.solver}.csv', index=False)
        
    explainer_source = shap.Explainer(source_only.predict_proba, X_stack)
    phi_S = explainer_source(X_stack).values
    explainer_target = shap.Explainer(target_only.predict_proba, X_stack)
    phi_T = explainer_target(X_stack).values
    delta_phi = phi_S - phi_T    # shape (n_samples, n_features)
    abs_gap   = np.abs(delta_phi).mean(0)
    sort_gap = np.sort(abs_gap)[::-1]
    for i_thr, sel_thr in enumerate(sort_gap):
        selected_shap = np.where(abs_gap >= sel_thr)[0]
        true = set(shift_subset_y)
        selected = set(selected_shap)
        tp = len(selected & true)  # True Positives
        fp = len(selected - true)  # False Positives
        fn = len(true - selected)  # False Negatives
        eval_results_shap = {'TP': tp, 'FP': fp, 'FN': fn, 'sel_thr': i_thr / len(sort_gap)}
        results_shap.append(eval_results_shap)
    df_results_shap = pd.DataFrame(results_shap)
    df_results_shap.to_csv(f'{directory}/shap_{args.model}_{args.solver}.csv', index=False)
    
    # wA, wB, new_X, new_weights = shared_reweight(X_S_test, X_T_test)
    eps = 1e-5
    proba_P2Q = np.clip(source_only.predict_proba(X_stack)[:, -1], eps, 1 - eps)
    proba_Q2P = np.clip(target_only.predict_proba(X_stack)[:, -1], eps, 1 - eps)
    new_Y = np.abs(proba_P2Q - proba_Q2P)

    region_model = DecisionTreeRegressor(max_depth=6,min_samples_leaf=100,min_samples_split=200,min_weight_fraction_leaf=0.05, ccp_alpha=0.0001, random_state=args.random_state+b).fit(X_stack, new_Y)# , sample_weight=new_weights)
    S = np.where(region_model.feature_importances_>0)[0]
    sel_whyshift = np.zeros(X_S.shape[1])
    sel_whyshift[S] = 1
    all_sel_whyshift.append(sel_whyshift)
    if b % 5 == 4:
        all_sel_whyshift = all_sel_whyshift[-5:]
        selfreq_whyshift = np.array(all_sel_whyshift).mean(axis = 0)
        for sel_thr in np.linspace(0.2, 1.0, 5):
            selected_whyshift = np.where(selfreq_whyshift >= sel_thr)[0]
            true = set(shift_subset_y)
            selected = set(selected_whyshift)
            tp = len(selected & true)  # True Positives
            fp = len(selected - true)  # False Positives
            fn = len(true - selected)  # False Negatives
            eval_results_whyshift = {'TP': tp, 'FP': fp, 'FN': fn, 'sel_thr': sel_thr}
            results_whyshift.append(eval_results_whyshift)
        df_results_whyshift = pd.DataFrame(results_whyshift)
        df_results_whyshift.to_csv(f'{directory}/whyshift_{args.model}_{args.solver}.csv', index=False)