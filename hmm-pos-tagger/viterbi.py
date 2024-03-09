# class HiddenMarkovModel 
import parse_treebank as pt 
import pandas as pd

class HiddenMarkovModel: 
    def __init__(self, treebank_dir, n_files): 
        self.all_docs = pt.read_all_treebank_files(n_files, treebank_dir)

        self.trans = pt.get_transition_matrix(self.all_docs)
        self.emit = pt.get_emission_matrix(self.all_docs)
        self.pi = pt.get_initial_state_matrix(self.all_docs)
        
        self.states = list(self.emit.index)
        self.vocab = list(self.emit.columns)

        self.n_states = len(self.states)
        self.n_vocab = len(self.vocab)
        
    def preprocess_obs(self, obs): 
        obs = "<s> " + obs + " <e>"
        return obs 
    
    def viterbi(self, obs): 
        proc_obs = self.preprocess_obs(obs)
        n_obs = len(proc_obs)
        obs_list = proc_obs.split(" ")
        V = pd.DataFrame(0, index=self.states, columns=obs_list)
        prev = pd.DataFrame(0, index=self.states, columns=obs_list)
        
        for s in range(self.n_states):
            tag = self.states[s]
            first_ob = obs_list[0]
            V.loc[tag, first_ob] = self.pi[tag] * self.emit.loc[tag, first_ob]

        for t in range(1, n_obs): 
            ob = obs_list[t]
            for s in range(self.n_states): 
                cur_state = self.states[s]
                for r in range(self.n_states): 
                    prev_state = self.states[r]
                    new_prob = V.loc[prev_state, ob] * self.trans.loc[prev_state, cur_state] * self.emit.loc[cur_state, ob]
                    if new_prob > V.loc[cur_state, ob]:
                        V.loc[cur_state, ob] = new_prob
                        prev.loc[cur_state, ob] = prev_state

        path = [0] * n_obs
        path[n_obs-1] = V.iloc[:, n_obs-1].idxmax()
        for t in range(n_obs-2, -1, -1): 
            path[t] = prev.loc[path[t+1], obs_list[t+1]]

        return path[1:-1]

# test HMM 
treebank_dir = "./treebank/tagged"
n_files = 199
hmm = HiddenMarkovModel(treebank_dir, n_files)
obs = "The vice director has went to New York"
path = hmm.viterbi(obs)
print(path)
