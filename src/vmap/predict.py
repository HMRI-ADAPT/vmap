"""
Viralmap
Version: v1.0
ADAPT (2025)
"""

# // imports
import torch
import numpy as np
import transformers
import pandas as pd
from .model import VMAPBase
from .visualize import VMAPViz
import warnings
from halo import Halo
import os

# // misc.
transformers.logging.set_verbosity_error()

# // inference class
class VMAPInfer():
    """
    Viralmap inference 
    """
    def __init__(self, vmap_base:str, vmap_mode:bool, device:str=None, model_name:str="facebook/esm2_t33_650M_UR50D"):
        # // vmap            
        if vmap_base is None: 
            raise ValueError("Path to base models directory missing.")
        
        # // device
        self.device = device if device else 'cpu' 
       
        # // feature columns
        self.feature_columns = ["feat_SP", "feat_CC", "feat_TM", "feat_NG", "feat_DR", "feat_FR", "feat_CH2", "feat_DB", "feat_CY", "feat_EX"]
        
        # // thresholds
        thresholds = np.array([[0.5,0.5],         # // SP  (HMM)
                               [0.5,0.5],         # // CC  (HMM)
                               [0.5,0.5],         # // TM  (HMM)
                               [0.5,0.13548404],  # // NG
                               [0.5,0.5],         # // DR  (HMM)
                               [0.5,0.13529132],  # // FR
                               [0.5,0.22461985],  # // CH2 (CHAIN)
                               [0.5,0.08119751],  # // DB
                               [0.5,0.5],         # // CY  (HMM)
                               [0.5,0.5]],        # // EX  (HMM)
                            dtype=np.float32)
        
        # // mode (thresholds, cooling)
        self.thresholds = thresholds[:,1]
        self.top_cool = 4.0 if vmap_mode else 1.0
        
        # // tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        
        # // base model
        spinner = Halo(text="Loading base model", spinner='star', color='cyan')
        spinner.start()
        self.models = []
        models_loaded = 0
        try:
            ckpt_paths = sorted([os.path.join(vmap_base, f) for f in os.listdir(vmap_base) if f.endswith('.pt')])
            for chkpt in ckpt_paths:
                models_loaded += 1
                spinner.text = f"Loading checkpoint: {models_loaded}/{len(ckpt_paths)}"
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=FutureWarning)
                    ckpt = torch.load(chkpt, map_location=self.device, weights_only=False)
                m = VMAPBase(model_name=model_name).to(self.device)
                m.load_state_dict(state_dict=ckpt['model_state_dict'], strict=False)
                m.eval()
                self.models.append(m)
            spinner.succeed(f"model weights loaded")
        except Exception as e:
            spinner.fail(f"{e}")
            raise
        
        # // hmm
        module_dir    = os.path.dirname(os.path.abspath(__file__))
        vmap_hmm_path = os.path.join(vmap_base, "vmap_hmm.npz")
        vmap_hmm      = np.load(vmap_hmm_path)
        for key in ['P_top', 'P_sp', 'P_cc', 'P_idr', 'A_top', 'A_sp', 'A_cc', 'A_idr']:
            setattr(self, key, torch.from_numpy(vmap_hmm[key]))

        # // rename
        self.rename_dict = {
                # // identifiers
                'Resid': 'Residue',
                'AA': 'AA',
                
                # // probability columns
                'feat_SP_prob': 'SP_prob',
                'feat_TM_prob': 'TM_prob',
                'feat_CY_prob': 'CY_prob',
                'feat_EX_prob': 'EX_prob',
                'feat_NG_prob': 'NG_prob',
                'feat_FR_prob': 'FR_prob',
                'feat_CH2_prob': 'CH_prob',
                'feat_DB_prob': 'DB_prob',
                'feat_CC_prob': 'CC_prob',
                'feat_DR_prob': 'DR_prob',
                
                # // final predictions (hmm & thresholds)
                'feat_SP_hmm': 'SP',
                'feat_TM_hmm': 'TM',
                'feat_CY_hmm': 'CY',
                'feat_EX_hmm': 'EX',
                'feat_NG_pred': 'NG',
                'feat_FR_pred': 'FR',
                'feat_CH2_pred': 'CH',
                'feat_DB_pred': 'DB',
                'feat_CC_hmm': 'CC',
                'feat_DR_hmm': 'DR'}
        
        # // visualization
        self.viz = VMAPViz()
        
    # // ################################################################## // 
    # // ############################ UTILS ############################### //
    # // ################################################################## // 
    @staticmethod
    def _summary_row(pname: str, summary: pd.DataFrame):
        """
        add to master summary sheet
        """
        row = {"Entry": pname}
        for _, r in summary.iterrows():
            row[r["Feature"]] = r["Regions / Positions"]
            row[f"{r['Feature']} (count)"] = r["Count"]
        return row


    # // ################################################################## // 
    # // ############################# HMM ################################ //
    # // ################################################################## // 
    @staticmethod
    def hmm_log_posterior(posteriors:np.ndarray, top_cool:float, eps:float=1e-12): 
        """
        logE
        neural network outputs are p(z_t | x_t), which are posteriors
        """
        assert posteriors.ndim == 2, "posteriors must be (L, K)"
        assert np.all((posteriors >= 0) & (posteriors <= 1)), "posteriors must be probabilities"

        # // add none column, and log
        s = posteriors.sum(axis=1, keepdims=True)   # // (L,1)
        none = 1.0 - s
        none = np.clip(none, 0.0, 1.0)
        probs = np.column_stack([none, posteriors]) # // (L,4)

        # // save
        log_post = np.log(np.clip(probs, eps, 1.0))
        logE = log_post

        # // logE for None cooling
        if logE.shape[1] > 2:
            logE[:, 0] -= top_cool # // topology
        else: 
            logE[:, 0] -= 1.0 # // others 
            
        # // log emissions
        return logE
    
    @staticmethod
    def hmm_viterbi(start_priors:np.ndarray, transition_matrix:np.ndarray, logE: np.ndarray, return_logprob: bool = False):
        """
        Viterbi decoding for an HMM-style structured decoder.
        Inputs:
            start_priors: (K,) start-state distribution over states
            transition_matrix: (K,K) row-stochastic transitions (from i to j)
            logE: (L,K) log emission potentials (e.g., log neural posteriors incl. NONE)
        Returns:
            path: (L,) most likely state sequence; optionally (path, best_logprob)
        """
        L, K = logE.shape

        # // log priors and log transitions
        # // keep forbidden transitions as -inf
        logP = np.log(np.clip(np.asarray(start_priors, dtype=np.float64), 1e-12, 1.0))          
        A = np.asarray(transition_matrix, dtype=np.float64)
        with np.errstate(divide='ignore'):
            logA = np.where(A > 0.0, np.log(A), -np.inf)                                 

        # // DP tables
        dp = np.full((L, K), -np.inf, dtype=np.float64)   # // best log-prob ending in state j at time t
        bp = np.full((L, K), -1, dtype=int)               # // backpointers

        # // init
        dp[0] = logP + logE[0]

        # // recurse
        for t in range(1, L):
            scores = dp[t-1][:, None] + logA                 # (K,K)
            best_prev = np.argmax(scores, axis=0)            # (K,)
            dp[t] = scores[best_prev, np.arange(K)] + logE[t]
            bp[t] = best_prev

        # // backtrack
        path = np.empty(L, dtype=int)
        path[-1] = int(np.argmax(dp[-1]))
        for t in range(L-2, -1, -1):
            path[t] = bp[t+1, path[t+1]]

        if return_logprob:
            return path, float(np.max(dp[-1]))
        return path
    
    
    def hmm_decode_one(self, name:str, cols:list, start_dist:np.ndarray, transition_matrix:np.ndarray, vmap_preds_csv:pd.DataFrame):
        """
        decode a single protein w.r.t single group with the hmm and add columns
        :param name: which hmm (TOP, SP, CC, DR)
        :param cols: which columns in viralmap prediction dataframe
        :param start_dist: start distribution (self.P_top, self.P_sp, self.P_cc, self.P_idr)
        :param transition_matrix: transition matrices (self.A_top, self.A_sp, self.A_cc, self.A_idr)
        :param vmap_preds_csv: output from vmap
        """
        posteriors = vmap_preds_csv[cols].to_numpy()
        logE       = self.hmm_log_posterior(posteriors=posteriors, top_cool=self.top_cool)

        path, _    = self.hmm_viterbi(start_priors       = start_dist, 
                                   transition_matrix  = transition_matrix, 
                                   logE               = logE, 
                                   return_logprob     = True)
        
        # // add new columns
        one_hot  = np.eye(len(cols)+1, dtype=np.uint8)[path]                              
        new_cols = [f"feat_NONE_{name}_hmm"]+[s.replace('_prob', '_hmm') for s in cols]
        vmap_preds_csv[new_cols] = one_hot
        return vmap_preds_csv
    

    def hmm_decode_protein(self, vmap_preds_csv:pd.DataFrame):
        """
        take in vmap output csv and return with hmm columns
        :param vmap_preds_csv: output from vmap
        """ 
        df_decode = self.hmm_decode_one(name              = "TOP", 
                                        cols              = ['feat_CY_prob','feat_EX_prob','feat_TM_prob'],
                                        start_dist        = self.P_top,
                                        transition_matrix = self.A_top, 
                                        vmap_preds_csv    = vmap_preds_csv)
        
        df_decode = self.hmm_decode_one(name              = "SP", 
                                        cols              = ['feat_SP_prob'],
                                        start_dist        = self.P_sp, 
                                        transition_matrix = self.A_sp, 
                                        vmap_preds_csv    = df_decode)
        
        df_decode = self.hmm_decode_one(name              = "CC", 
                                        cols              = ['feat_CC_prob'],
                                        start_dist        = self.P_cc,
                                        transition_matrix = self.A_cc, 
                                        vmap_preds_csv    = df_decode)
        
        df_decode = self.hmm_decode_one(name              = "DR", 
                                        cols              = ['feat_DR_prob'],
                                        start_dist        = self.P_idr,
                                        transition_matrix = self.A_idr, 
                                        vmap_preds_csv    = df_decode)
        # // 
        assert (df_decode[["feat_NONE_TOP_hmm","feat_CY_hmm","feat_EX_hmm","feat_TM_hmm"]].sum(1) == 1).all()
        return df_decode
    
    # // ################################################################## // 
    # // ########################### INFERENCE ############################ //
    # // ################################################################## // 
    def predict_one(self, seq:str, pname:str, output_dir:str):
        """
        run prediction for one protein
        """
        # // tokenize sequence
        tokens = self.tokenizer(seq, return_tensors="pt", padding=False)
        tokens = {k: v.to(self.device) for k,v in tokens.items()}

        # // logits
        with torch.no_grad():
            all_logits = torch.stack([m(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask']) for m in self.models]) 
            logits = all_logits.mean(dim=0)  
            
        # // predictions
        logits_trunc = logits[0:, 1:-1, :].squeeze(0)
        probs        = torch.sigmoid(logits_trunc).cpu().numpy()
        preds        = (probs >= self.thresholds).astype(np.uint8)

        # // format df
        res_no = np.arange(1, len(seq)+1)
        df = pd.DataFrame({
            "Resid":   res_no,
            "AA":      list(seq)
        })
        for i, feat in enumerate(self.feature_columns):
            df[f"{feat}_prob"] = probs[:, i]
            df[f"{feat}_pred"] = preds[:, i]
        
        # // hmm
        df_hmm = self.hmm_decode_protein(vmap_preds_csv=df)

        # // cleanup
        df_out = df_hmm[list(self.rename_dict.keys())].rename(columns=self.rename_dict)

        # // save
        df_out.to_csv(f"{output_dir}/{pname}_vmap_preds.csv", index=False)
        _, summary = self.viz.render_interactive(df=df_out, save_out=output_dir, name=pname, title=pname)
        return summary


    def predict(self, file_path:str, output_dir:str):
        """
        fasta file -> directory of sequences
        """
        summary_rows = []

        # // count total proteins, not total lines
        total_proteins = 0
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith(">"):
                    total_proteins += 1
        
        spinner = Halo(text=f'Generating annotations [w/ {self.device}]:', spinner='star', color='cyan')
        spinner.start()

        current_header     = None
        current_sequence   = []
        proteins_processed = 0

        with open(file_path, 'r') as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                
                if line.startswith(">"):
                    # // process previous entry before starting new one
                    if current_header is not None:
                        sequence = ''.join(current_sequence)
                        proteins_processed += 1
                        
                        # // update
                        spinner.text = f'Generating annotations [w/ {self.device}]: {proteins_processed}/{total_proteins} proteins ({proteins_processed/total_proteins*100:.2f}%)'
                        
                        # // predict
                        summary = self.predict_one(seq=sequence, pname=current_header, output_dir=output_dir)

                        # // save to overview
                        summary_rows.append(self._summary_row(current_header, summary))

                                            
                    # // start new entry
                    current_header = line[1:].strip()
                    current_sequence = []
                    
                elif line.strip() == "":
                    continue
                else:
                    sequence_line = line.replace(" ", "").upper()
                    current_sequence.append(sequence_line)
            
            # // process the last protein
            if current_header is not None:
                sequence = ''.join(current_sequence)
                proteins_processed += 1
                spinner.text = f'Generating annotations: {proteins_processed}/{total_proteins} proteins (100.00%)'
                summary = self.predict_one(seq=sequence, pname=current_header, output_dir=output_dir)
                summary_rows.append(self._summary_row(current_header, summary))


        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)        
        spinner.succeed(f'Annotations generated for {proteins_processed} proteins.')

   





                



        


        
  
    


        






        












        

        


