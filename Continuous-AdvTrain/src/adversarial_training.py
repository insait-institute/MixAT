from typing import Dict, Tuple
import logging
import torch

from contextlib import contextmanager, nullcontext

from transformers import (
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers.integrations.integration_utils import TensorBoardCallback
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DPOTrainer
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from contextlib import nullcontext

import embedding_attack
import model_utils
import data

import gc
import numpy as np
import copy

def adversarial_training_loop(
    model_name,
    path_config,
    adversarial_training_config,
    dataset_config,
    training_config,
    peft_config,
    bnb_config,
    sfttrainer_config,
    trainer_hparams,
):
    schedule_to_use = training_config["schedule"]
    training_config.pop("schedule")
    continue_training = training_config["resume_from_checkpoint"] is not None
    db_continue = training_config["db_continue"]
    training_config.pop("db_continue")
    with_lora_adapter =  training_config["with_lora_adapter"]
    training_config.pop("with_lora_adapter")
    with_kl_loss =  training_config["with_kl_loss"]
    training_config.pop("with_kl_loss")
    kl_constant =  training_config["kl_constant"]
    training_config.pop("kl_constant")
    is_static_training = training_config["is_static_training"]
    training_config.pop("is_static_training")


    
    # ======= Load model and tokeinzer ======= #
    if bnb_config is not None:
        bnb_config = BitsAndBytesConfig(**bnb_config)
    
    if not with_lora_adapter:
        bnb_config = None
    
    model, tokenizer = model_utils.load_model_and_tokenizer(
        path_config["model_path"],
        bnb_config=bnb_config,
        padding_side=trainer_hparams["padding_side"],
        dtype=trainer_hparams.pop("dtype"),
    )



    if trainer_hparams["do_online_dpo"]:
        reference_model = model_utils.load_model_and_tokenizer(
            path_config["model_path"], bnb_config=bnb_config, padding_side=trainer_hparams["padding_side"]
        )[0]
        reference_model.eval()
    else:
        reference_model = None

    # ======= Load Data ======= #
    train_data, val_data = data.load_adversarial_training_data(
        data_path=dataset_config["data_path"],
        utility_data=dataset_config["utility_data"],
        probabilities=dataset_config["probabilities"],
        model_name=model_name,
        tokenizer=tokenizer,
        stopping_strategy=dataset_config["stopping_strategy"],
        diverse_safe_answers=dataset_config["diverse_safe_answers"],
        restricted_trainingset_size=dataset_config["restricted_trainingset_size"],
        seed_for_interleave_dataset = dataset_config["seed_for_interleave_dataset"]
    )
    df = train_data.to_pandas()
    df = df[df["dataset_id"] == 1]
    
    logging.info(f"Loaded training data with {len(train_data)} samples")

    # ======= Set Formatting Prompts Function ======= #
    formatting_func, collator, response_key = data.get_prompt_formatting_func_and_collator(
        model_name, tokenizer
    )
    if not with_lora_adapter:
        peft_config = None

    # ======= Set Lora Config ======= #
    if peft_config is not None and isinstance(peft_config["target_modules"], str) is False:
        peft_config["target_modules"] = list(peft_config["target_modules"])
    if with_lora_adapter:
        peft_config = LoraConfig(**peft_config)
    else:
        peft_config = None
    

    # ======= Set Training Arguments ======= #
    training_arguments = TrainingArguments(
        **training_config,
        output_dir=path_config["logging_path"] + "/trainer_output",
        logging_dir=path_config["logging_path"] + "/trainer_logs",
        #report_to="none"
        # gradient_checkpointing=True,
        # gradient_checkpointing_kwargs={"use_reentrant":False},
    )

    

    # ======= Set Trainer ======= #
    if peft_config is not None:
        trainer_config = {
            "model": model,
            "train_dataset": train_data,
            "eval_dataset": val_data,
            "formatting_func": formatting_func,
            "data_collator": collator,
            "peft_config": peft_config,
            "tokenizer": tokenizer,
            "args": training_arguments,
            "packing": sfttrainer_config["packing"],
            "max_seq_length": sfttrainer_config["max_seq_length"],
            "dpo_reference_model": reference_model,
            **trainer_hparams,
        }
    else:
        trainer_config = {
            "model": model,
            "train_dataset": train_data,
            "eval_dataset": val_data,
            "formatting_func": formatting_func,
            "data_collator": collator,
            "tokenizer": tokenizer,
            "args": training_arguments,
            "packing": sfttrainer_config["packing"],
            "max_seq_length": sfttrainer_config["max_seq_length"],
            "dpo_reference_model": reference_model,
            **trainer_hparams,
        }
    

    log_hparams = {
        "learning_rate": training_arguments.learning_rate,
        **trainer_hparams,
        **path_config,
    }
    if peft_config is not None:
        log_hparams = {**log_hparams, "target_modules": str(peft_config.target_modules)}
   
    if trainer_hparams["trainer_type"] == "ul":
        if (tokenizer.unk_token is None):
          
            if "llama3.1" in model_name or "llama3" in model_name:
                tokenizer.pad_token = '<|reserved_special_token_31|>'
            elif tokenizer.pad_token is not None:
                pass
            else:
                raise ValueError("No token for padding")

        elif "mistral" in model_name.lower():    
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("No token for padding")
        
        if "mistral" in model_name.lower():
            is_zero_a_valid_index = False
        else:
            is_zero_a_valid_index = True

        tokenizer.truncation_side = 'right'
        tokenizer.padding_side = 'left'


        # ====== Init Attack ======= #

        embed_weights = model_utils.get_embed_weights(model)
        attack_type = adversarial_training_config.pop("attack_type")

        
        
        if attack_type == "NoAttack":
            adversarial_attack = embedding_attack.NoAttack(
                embed_weights,
                is_zero_a_valid_index=is_zero_a_valid_index
            )
        elif attack_type == "mixed_pap":
            adversarial_attack = embedding_attack.MixedAttackPAP(
                embed_weights,
                response_key=response_key,
                tokenizer=tokenizer,
                model_path = path_config["model_path"],
                is_zero_a_valid_index=is_zero_a_valid_index,
                **adversarial_training_config,
            )
        elif attack_type == "mixed_perturbed":
            adversarial_attack = embedding_attack.MixedAttackPerturbed(
                embed_weights,
                response_key=response_key,
                tokenizer=tokenizer,
                model_path = path_config["model_path"],
                is_zero_a_valid_index=is_zero_a_valid_index,
                **adversarial_training_config,
            )
        elif attack_type == "mixed_perturbed_with_GCG":
             adversarial_attack = embedding_attack.MixedAttackPerturbedPAPGCG(
                embed_weights,
                response_key=response_key,
                tokenizer=tokenizer,
                model_path = path_config["model_path"],
                is_zero_a_valid_index=is_zero_a_valid_index,
                **adversarial_training_config,
            )
        elif attack_type == "mixed_perturbed_llama3_with_GCG":
             adversarial_attack = embedding_attack.MixedAttackPerturbedLLAMAPAPGCG(
                embed_weights,
                response_key=response_key,
                tokenizer=tokenizer,
                model_path = path_config["model_path"],
                is_zero_a_valid_index=is_zero_a_valid_index,
                **adversarial_training_config,
            )
        elif attack_type == "mixed_perturbed_qwen_with_GCG":
             adversarial_attack = embedding_attack.MixedAttackPerturbedQWENPAPGCG(
                embed_weights,
                response_key=response_key,
                tokenizer=tokenizer,
                model_path = path_config["model_path"],
                is_zero_a_valid_index=is_zero_a_valid_index,
                **adversarial_training_config,
            )
        elif attack_type == "mixed_perturbed_qwen":
            adversarial_attack = embedding_attack.MixedAttackPerturbedQWEN(
                embed_weights,
                response_key=response_key,
                tokenizer=tokenizer,
                model_path = path_config["model_path"],
                is_zero_a_valid_index=is_zero_a_valid_index,
                **adversarial_training_config,
            )
        elif attack_type == "mixed_perturbed_llama3_1":
            adversarial_attack = embedding_attack.MixedAttackPerturbedLLAMA3_1(
                embed_weights,
                response_key=response_key,
                tokenizer=tokenizer,
                model_path = path_config["model_path"],
                is_zero_a_valid_index=is_zero_a_valid_index,
                **adversarial_training_config,
            )
        elif attack_type == "mixed_perturbed_no_attack_llama3_1":
            adversarial_attack = embedding_attack.MixedAttackPerturbedNoAttackLLAMA3_1(
                embed_weights,
                response_key=response_key,
                tokenizer=tokenizer,
                model_path = path_config["model_path"],
                is_zero_a_valid_index=is_zero_a_valid_index,
                **adversarial_training_config,
            )
        elif attack_type == "mixed_perturbed_no_attack_qwen":
            adversarial_attack = embedding_attack.MixedAttackPerturbedNoAttackQWEN(
                embed_weights,
                response_key=response_key,
                tokenizer=tokenizer,
                model_path = path_config["model_path"],
                is_zero_a_valid_index=is_zero_a_valid_index,
                **adversarial_training_config,
            )
        elif attack_type == "mixed_perturbed_no_attack_mistral":
            adversarial_attack = embedding_attack.MixedAttackPerturbedNoAttackMistral(
                embed_weights,
                response_key=response_key,
                tokenizer=tokenizer,
                model_path = path_config["model_path"],
                is_zero_a_valid_index=is_zero_a_valid_index,
                **adversarial_training_config,
            )
        elif attack_type == "mixed_perturbed_mistral":
            adversarial_attack = embedding_attack.MixedAttackPerturbedMistral(
                embed_weights,
                response_key=response_key,
                tokenizer=tokenizer,
                model_path = path_config["model_path"],
                is_zero_a_valid_index=is_zero_a_valid_index,
                **adversarial_training_config,
            )
        elif attack_type == "mixed_perturbed_dpo":
            adversarial_attack = embedding_attack.MixedAttackPerturbedDPO(
                embed_weights,
                response_key=response_key,
                tokenizer=tokenizer,
                model_path = path_config["model_path"],
                is_zero_a_valid_index=is_zero_a_valid_index,
                **adversarial_training_config,
        )
        elif attack_type == "embedding":
            adversarial_attack = embedding_attack.EmbeddingSpaceAttack(
                embed_weights,
                response_key=response_key,
                tokenizer=tokenizer,
                is_zero_a_valid_index=is_zero_a_valid_index,
                **adversarial_training_config,
                )
        elif attack_type == "mixed_perturbed_llama": 
            adversarial_attack = embedding_attack.MixedAttackPerturbedLLAMA(
                embed_weights,
                response_key=response_key,
                tokenizer=tokenizer,
                model_path = path_config["model_path"],
                is_zero_a_valid_index=is_zero_a_valid_index,
                **adversarial_training_config,
            )
        elif attack_type == "mixed_perturbed_no_attack_llama3": 
            adversarial_attack = embedding_attack.MixedAttackPerturbedNoAttackLLAMA(
                embed_weights,
                response_key=response_key,
                tokenizer=tokenizer,
                model_path = path_config["model_path"],
                is_zero_a_valid_index=is_zero_a_valid_index,
                **adversarial_training_config,
            )
        elif attack_type == "mixed_perturbed_llama2": 
            adversarial_attack = embedding_attack.MixedAttackPerturbedLLAMA2(
                embed_weights,
                response_key=response_key,
                tokenizer=tokenizer,
                model_path = path_config["model_path"],
                is_zero_a_valid_index=is_zero_a_valid_index,
                **adversarial_training_config,
            )
        elif attack_type == "mixed_perturbed_no_attack_llama2":
            adversarial_attack = embedding_attack.MixedAttackPerturbedNoAttackLLAMA2(
                embed_weights,
                response_key=response_key,
                tokenizer=tokenizer,
                model_path = path_config["model_path"],
                is_zero_a_valid_index=is_zero_a_valid_index,
                **adversarial_training_config,
            )
        else:
            raise "No such attack"
        
        scheduling = None
        #I use seed 42 when creating the dataset, which determines the number of steps, if epoch count is 5
        #760*8 = 6080 

        if schedule_to_use == "uni5":
            r_arr = np.random.permutation(6080)
            scheduling = np.zeros((6080,),dtype=np.float32)
            scheduling[r_arr[:int(6080/20)]] = 1.0
            assert np.sum(scheduling) == 304,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uni10":
            r_arr = np.random.permutation(6080)
            scheduling = np.zeros((6080,),dtype=np.float32)
            scheduling[r_arr[:int(6080/10)]] = 1.0
            assert np.sum(scheduling) == 608.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uni20":
            r_arr = np.random.permutation(6080)
            scheduling = np.zeros((6080,),dtype=np.float32)
            scheduling[r_arr[:int(6080/5)]] = 1.0
            assert np.sum(scheduling) == 1216.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uni30":
            r_arr = np.random.permutation(6080)
            scheduling = np.zeros((6080,),dtype=np.float32)
            scheduling[r_arr[:int(3*6080/10)]] = 1.0
            assert np.sum(scheduling) == 1824.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uni40":
            r_arr = np.random.permutation(6080)
            scheduling = np.zeros((6080,),dtype=np.float32)
            scheduling[r_arr[:int(4*6080/10)]] = 1.0
            assert np.sum(scheduling) == 2432.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uni50": #ALLITOTTAM
            r_arr = np.random.permutation(6040)
            scheduling = np.zeros((6040,),dtype=np.float32)
            scheduling[r_arr[:int(5*6040/10)]] = 1.0
            assert np.sum(scheduling) == 3020.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uniform50_more":
            scheduling = []
            r_arr = np.random.permutation(8512)
            scheduling = np.zeros((8512,),dtype=np.float32)
            scheduling[r_arr[:int(5*8512/10)]] = 1.0
            assert np.sum(scheduling) == 4256.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uniform50_epoch2":
            scheduling = []
            r_arr = np.random.permutation(2416)
            scheduling = np.zeros((2416,),dtype=np.float32)
            scheduling[r_arr[:int(5*2416/10)]] = 1.0
            assert np.sum(scheduling) == 1208.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uniform45_10_epoch2": 
            scheduling = []
            r_arr = np.random.permutation(2416)
            scheduling = np.zeros((2416,),dtype=np.float32)
            scheduling[r_arr[:int(45*2416/100)]] = 1.0
            scheduling[r_arr[int(45*2416/100):int(55*2416/100)]] = 2.0
            assert np.sum(scheduling==1.0) == 1087.0,"Wrong percent"
            assert np.sum(scheduling==2.0) == 241.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uniform5_epoch2":
            scheduling = []
            r_arr = np.random.permutation(2432)
            scheduling = np.zeros((2432,),dtype=np.float32)
            scheduling[r_arr[:int(5*2432/100)]] = 1.0
            #assert np.sum(scheduling) == 1216.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uniform10_epoch2":
            scheduling = []
            r_arr = np.random.permutation(2432)
            scheduling = np.zeros((2432,),dtype=np.float32)
            scheduling[r_arr[:int(10*2432/100)]] = 1.0
            #assert np.sum(scheduling) == 1216.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uniform20_epoch2":
            scheduling = []
            r_arr = np.random.permutation(2432)
            scheduling = np.zeros((2432,),dtype=np.float32)
            scheduling[r_arr[:int(20*2432/100)]] = 1.0
            #assert np.sum(scheduling) == 1216.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uniform30_epoch2":
            scheduling = []
            r_arr = np.random.permutation(2432)
            scheduling = np.zeros((2432,),dtype=np.float32)
            scheduling[r_arr[:int(30*2432/100)]] = 1.0
            #assert np.sum(scheduling) == 1216.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uniform60_epoch2":
            scheduling = []
            r_arr = np.random.permutation(2432)
            scheduling = np.zeros((2432,),dtype=np.float32)
            scheduling[r_arr[:int(60*2432/100)]] = 1.0
            #assert np.sum(scheduling) == 1216.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uniform70_epoch2":
            scheduling = []
            r_arr = np.random.permutation(2432)
            scheduling = np.zeros((2432,),dtype=np.float32)
            scheduling[r_arr[:int(70*2432/100)]] = 1.0
            #assert np.sum(scheduling) == 1216.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uniform80_epoch2":
            scheduling = []
            r_arr = np.random.permutation(2432)
            scheduling = np.zeros((2432,),dtype=np.float32)
            scheduling[r_arr[:int(80*2432/100)]] = 1.0
            #assert np.sum(scheduling) == 1216.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uniform90_epoch2":
            scheduling = []
            r_arr = np.random.permutation(2432)
            scheduling = np.zeros((2432,),dtype=np.float32)
            scheduling[r_arr[:int(90*2432/100)]] = 1.0
            #assert np.sum(scheduling) == 1216.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uniform100_epoch2": 
            scheduling = []
            r_arr = np.random.permutation(2416)
            scheduling = np.ones((2416,),dtype=np.float32)
            #assert np.sum(scheduling) == 1216.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uniform40_epoch2":
            scheduling = []
            r_arr = np.random.permutation(2432)
            scheduling = np.zeros((2432,),dtype=np.float32)
            scheduling[r_arr[:int(40*2432/100)]] = 1.0
            #assert np.sum(scheduling) == 1216.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uni60":
            r_arr = np.random.permutation(6080)
            scheduling = np.zeros((6080,),dtype=np.float32)
            scheduling[r_arr[:int(6*6080/10)]] = 1.0
            assert np.sum(scheduling) == 3648.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uni8_dpo":
            r_arr = np.random.permutation(2880) #8*360
            scheduling = np.zeros((2880,),dtype=np.float32)
            scheduling[r_arr[:int(235)]] = 1.0
            assert np.sum(scheduling) == 235.0,"Wrong percent"
            scheduling = scheduling.tolist()
        elif schedule_to_use == "uniform8":
            scheduling = []
            for i in range(6080):
                scheduling.append(0.082)
        elif schedule_to_use == "uniform50":
            scheduling = []
            for i in range(6080):
                scheduling.append(0.5)
        elif schedule_to_use == "uniform100":
            scheduling = []
            for i in range(6080):
                scheduling.append(1.0)
        elif schedule_to_use == "beginning_concentrated":
            scheduling = []
            for i in range(80):
                scheduling.append(0.8)
            for i in range(80):
                scheduling.append(0.5)
            for i in range(300):
                scheduling.append(0.2)
            for i in range(1000):
                scheduling.append(0.1)
            for i in range(4620):
                scheduling.append(0.05)
        elif schedule_to_use == "end_concentrated":
            scheduling = []
            for i in range(4620):
                scheduling.append(0.05)
            for i in range(1000):
                scheduling.append(0.1)
            for i in range(300):
                scheduling.append(0.2)
            for i in range(80):
                scheduling.append(0.5)
            for i in range(80):
                scheduling.append(0.8)
        elif schedule_to_use is not None:
            raise "No such a schedule"
    
        trainer = AdversarialULTrainer(
            adversarial_attack=adversarial_attack,
            embed_weights=embed_weights,
            hparams=log_hparams,
            checkpointing_my_path=path_config["checkpoint_path"],
            scheduling=scheduling,
            db_continue = db_continue,
            is_zero_a_valid_index=is_zero_a_valid_index,
            kl_constant = kl_constant,
            is_static_training = is_static_training,
            with_kl_loss = with_kl_loss,
            **trainer_config,
        )
 
    elif trainer_hparams["trainer_type"] == "dpo":
        scheduling = None
        if schedule_to_use == "uni8_dpo":
            r_arr = np.random.permutation(2880) #8*360
            scheduling = np.zeros((2880,),dtype=np.float32)
            scheduling[r_arr[:int(235)]] = 1.0
            assert np.sum(scheduling) == 235.0,"Wrong percent"
            scheduling = scheduling.tolist()
        tokenizer.pad_token = tokenizer.unk_token
        trainer = AdversarialDPOTrainer(
            adversarial_attack=adversarial_attack,
            embed_weights=embed_weights,
            hparams=log_hparams,
            scheduling=scheduling,
            db_continue = db_continue,
            is_zero_a_valid_index=is_zero_a_valid_index,
            **trainer_config,
        )

    # ======= Train ======= #
    if trainer_hparams["restart_count"] > 0:
        trainer.train(resume_from_checkpoint=True)
    else:
        if continue_training:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

    # ======= Save Model ======= #
    trainer.model.save_pretrained(path_config["checkpoint_path"] + "/final_model")


class AdversarialULTrainer(SFTTrainer):
    def __init__(
        self,
        adversarial_attack,
        embed_weights,
        hparams,
        checkpointing_my_path,
        away_weight=1,
        toward_weight=1,
        utility_weight=1,
        ema_weight=0,
        away_cutoff=-10000,
        toward_cutoff=0,
        away_loss_type="negative_cross_entropy",
        do_online_dpo=False,
        dpo_loss_type="ipo",
        dpo_beta=0.1,
        dpo_weight=1.0,
        dpo_reference_model=None,
        dpo_label_smoothing=0.0,
        scheduling=None,
        db_continue = 0,
        is_zero_a_valid_index=False,
        with_kl_loss = False,
        kl_constant = None,
        is_static_training = False,
        *args,
        **kwargs,
    ):
        # pop arguments not used by the parent class
        invalid_args = {
            "dpo_loss_type",
            "dpo_beta",
            "dpo_weight",
            "padding_side",
            "restart_count",
            "trainer_type",
            "dpo_label_smoothing",
        }
        for arg in invalid_args:
            kwargs.pop(arg, None)
        super().__init__(*args, **kwargs)

        self.db = db_continue
        self.scheduling  = scheduling
   

        self.is_zero_a_valid_index=is_zero_a_valid_index
        self.is_static_training = is_static_training

        self.static_model = None

        self.checkpoint_path = checkpointing_my_path
        
        self.adversarial_attack = adversarial_attack
        self.embed_weights = embed_weights
        self.hparams = hparams
        self.away_weight = away_weight
        self.toward_weight = toward_weight
        self.utility_weight = utility_weight
        self.ema_weight = ema_weight
        self.away_loss_ema = None
        self.toward_loss_ema = None
        self.utility_loss_ema = None
        self.away_cutoff = away_cutoff
        self.toward_cutoff = toward_cutoff
        self.away_loss_type = away_loss_type

        self.with_kl_loss = with_kl_loss
        self.kl_constant = kl_constant


        # DPO
        self.do_online_dpo = do_online_dpo
        self.dpo_loss_type = dpo_loss_type
        self.beta = dpo_beta
        self.dpo_weight = dpo_weight
        self.label_smoothing = dpo_label_smoothing
        # hard-coded DPO
        self.precompute_ref_log_probs = True
        self._precomputed_train_ref_log_probs = False
        self.is_encoder_decoder = False
        self._peft_has_been_casted_to_bf16 = False
     
        self.reference_free = False
        self.label_pad_token_id = -100
        self.dpo_reference_model = dpo_reference_model

    def compute_loss(self, model, inputs, return_outputs=False):
        self.db = self.db +1       

        if self.is_static_training and self.static_model is None:
            self.static_model = copy.deepcopy(model)
            self.static_model.eval()
 
 
        # inputs keys (['input_ids', 'attention_mask', 'labels', 'dataset_id'])
        toward_inputs, away_inputs, utility_inputs = self.split_inputs(inputs)

        away_loss, toward_loss, utility_loss, dpo_loss, attack_loss = (
            torch.tensor(0, device=model.device),
            torch.tensor(0, device=model.device),
            torch.tensor(0, device=model.device),
            torch.tensor(0, device=model.device),
            torch.tensor(0, device=model.device),
        )
        away_text, utility_text = "", ""
        attack_losses, affirmative_responses = [], []

        # ======= away loss =======
        if away_inputs is not None:
            if self.scheduling is not None:
                temp_ratio = self.scheduling[self.db-1]
            else:
                temp_ratio = 1.0
            if self.is_static_training:
                    input_embeds, adv_perturbation, adv_perturbation_mask, attack_losses, affirmative_responses, is_gcg = (
                    self.adversarial_attack.attack(
                        self.static_model,
                        away_inputs["input_ids"],
                        away_inputs["labels"],
                        away_inputs["attention_mask"],
                        ratio = temp_ratio
                    )
                )
            else:
                input_embeds, adv_perturbation, adv_perturbation_mask, attack_losses, affirmative_responses, is_gcg = (
                    self.adversarial_attack.attack(
                        model,
                        away_inputs["input_ids"],
                        away_inputs["labels"],
                        away_inputs["attention_mask"],
                        ratio = temp_ratio
                    )
                )

            if len(attack_losses) > 0:
                attack_loss = sum(attack_losses) / len(attack_losses)
            perturbted_inputs_embeds_away = self.adversarial_attack.get_adv_embeddings(
                input_embeds, adv_perturbation, adv_perturbation_mask, is_gcg
            )
            outputs = model(
                inputs_embeds=perturbted_inputs_embeds_away,
                attention_mask=away_inputs["attention_mask"],
                labels=away_inputs["labels"],
            )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            away_logits = outputs[1]
            away_text = repr(
                model_utils.logits_to_text(
                    away_logits[0, (away_inputs["labels"][0] != -100).nonzero().squeeze() - 1].unsqueeze(0),
                    self.tokenizer,
                )
            )
            if self.away_loss_type == "negative_cross_entropy":
                if self.away_cutoff > -outputs[0]:
                    away_loss = self.away_cutoff + 0.001 * (-outputs[0])
                else:
                    away_loss = -outputs[0]
            elif self.away_loss_type == "log_1_minus_p":
                away_loss = log_1_minus_p_loss(
                    away_logits[:, :-1], away_inputs["labels"][:, 1:], self.away_cutoff
                )

            if self.do_online_dpo:
                away_logps = DPOTrainer.get_batch_logps(
                    away_logits,
                    away_inputs["labels"],
                    average_log_prob=self.dpo_loss_type == "ipo",
                    is_encoder_decoder=False,
                    label_pad_token_id=self.label_pad_token_id,
                )
                with torch.no_grad():
                    away_ref_logits = self.dpo_reference_model(
                        inputs_embeds=perturbted_inputs_embeds_away.to(self.dpo_reference_model.dtype),
                        attention_mask=away_inputs["attention_mask"],
                    )[0]
                    away_ref_logps = DPOTrainer.get_batch_logps(
                        away_ref_logits,
                        away_inputs["labels"],
                        average_log_prob=self.dpo_loss_type == "ipo",
                        is_encoder_decoder=False,
                        label_pad_token_id=self.label_pad_token_id,
                    )

        # ======= toward loss =======

        if toward_inputs is not None:
            if is_gcg:
                for i in range(toward_inputs["input_ids"].shape[0]):
                    if self.is_zero_a_valid_index: 
                        target_mask_away = away_inputs["labels"][i] >= 0
                    else:
                        target_mask_away = away_inputs["labels"][i] > 0
   
                    input_mask_away = (~target_mask_away * away_inputs["attention_mask"][i]).to(bool)
                    if self.is_zero_a_valid_index: 
                        target_mask_toward = toward_inputs["labels"][i] >= 0
                    else:
                        target_mask_toward = toward_inputs["labels"][i] > 0
 
                    input_mask_toward = (~target_mask_toward * toward_inputs["attention_mask"][i]).to(bool)
                    temp = away_inputs["input_ids"][i,input_mask_away]

                    index = torch.nonzero(input_mask_toward).squeeze(-1)[-1].item() #TODO
                    toward_inputs["input_ids"][i,index+1-temp.shape[0]:index+1] = temp
                    toward_inputs["attention_mask"][i,index+1-temp.shape[0]:index+1] = 1
                   
            perturbted_inputs_embeds_toward = self.get_away_perturbation_from_toward_perturbation(
                toward_inputs, adv_perturbation, adv_perturbation_mask
            )
            outputs = model(
                inputs_embeds=perturbted_inputs_embeds_toward,
                attention_mask=toward_inputs["attention_mask"],
                labels=toward_inputs["labels"],
            )
            toward_logits = outputs[1]
            if self.toward_cutoff > outputs[0]:
                toward_loss = self.toward_cutoff + 0.001 * (outputs[0])
            else:
                toward_loss = outputs[0]

            if self.do_online_dpo:
                toward_logps = DPOTrainer.get_batch_logps(
                    toward_logits,
                    toward_inputs["labels"],
                    average_log_prob=self.dpo_loss_type == "ipo",
                    is_encoder_decoder=False,
                    label_pad_token_id=self.label_pad_token_id,
                )
                with torch.no_grad():
                    toward_ref_logits = self.dpo_reference_model(
                        inputs_embeds=perturbted_inputs_embeds_toward.to(self.dpo_reference_model.dtype),
                        attention_mask=toward_inputs["attention_mask"],
                    )[0]
                    toward_ref_logps = DPOTrainer.get_batch_logps(
                        toward_ref_logits,
                        toward_inputs["labels"],
                        average_log_prob=self.dpo_loss_type == "ipo",
                        is_encoder_decoder=False,
                        label_pad_token_id=self.label_pad_token_id,
                    )

        # ======= utility loss =======
        if utility_inputs is not None:
            outputs = model(
                input_ids=utility_inputs["input_ids"],
                attention_mask=utility_inputs["attention_mask"],
                labels=utility_inputs["labels"],
            )
            utility_logits = outputs[1]
            utility_text = repr(
                model_utils.logits_to_text(
                    utility_logits[
                        0, (utility_inputs["labels"][0] != -100).nonzero().squeeze() - 1
                    ].unsqueeze(0),
                    self.tokenizer,
                )
            )
    
            utility_loss = outputs[0]
    
            
            if self.with_kl_loss:
                assert isinstance(model, PeftModel), "The model must be a peft_model to run KL-penalty"
                with torch.autocast(device_type="cuda"):
                    with torch.no_grad():
                        model.disable_adapter_layers()
                        base_logits = model(input_ids=utility_inputs["input_ids"],).logits
                        base_logits = base_logits[utility_inputs["attention_mask"]].log_softmax(dim=-1)
                        model.enable_adapter_layers()
                    new_logits = model(input_ids=utility_inputs["input_ids"]).logits
                    new_logits = new_logits[utility_inputs["attention_mask"]].softmax(dim=-1)
                    kl_loss = F.kl_div(base_logits, new_logits)
                kl_loss = kl_loss / (kl_loss.detach() + 1e-8)
                (self.kl_constant * kl_loss).backward()
   

        if self.do_online_dpo:
            if away_inputs is not None:
                dpo_loss = get_dpo_loss(self, toward_logps, away_logps, toward_ref_logps, away_ref_logps)[
                    0
                ].mean()
            loss = self.dpo_weight * dpo_loss + self.utility_weight * utility_loss
        else:
            # TODO Remove does not do anything anyway (we do not retain grads)
            if self.ema_weight > 0:
                self.toward_loss_ema = (
                    self.ema_weight * self.toward_loss_ema + (1 - self.ema_weight) * toward_loss
                    if self.toward_loss_ema is not None
                    else toward_loss
                )
                self.away_loss_ema = (
                    self.ema_weight * self.away_loss_ema + (1 - self.ema_weight) * away_loss
                    if self.away_loss_ema is not None
                    else away_loss
                )
                self.utility_loss_ema = (
                    self.ema_weight * self.utility_loss_ema + (1 - self.ema_weight) * utility_loss
                    if self.utility_loss_ema is not None
                    else utility_loss
                )
                loss = (
                    self.away_weight * self.away_loss_ema
                    + self.toward_weight * self.toward_loss_ema
                    + self.utility_weight * self.utility_loss_ema
                )
            else:
                loss = (
                    self.away_weight * away_loss
                    + self.toward_weight * toward_loss
                    + self.utility_weight * utility_loss
                )

        log(
            self,
            loss,
            away_loss,
            toward_loss,
            dpo_loss,
            utility_loss,
            attack_losses,
            attack_loss,
            affirmative_responses,
            away_text,
            utility_text,
        )

        del outputs
        del toward_inputs
        del away_inputs
        torch.cuda.empty_cache()
        gc.collect()
        return loss

    def split_inputs(self, inputs):
        unique_ids = list(torch.unique(inputs["dataset_id"]).int().cpu().numpy())
        splits = {}
        for id in unique_ids:
            mask = inputs["dataset_id"] == id
            label_subset = {k: v[mask] for k, v in inputs.items()}
            splits[id] = label_subset

        away_inputs, toward_inputs, utility_inputs = None, None, None

        if 0 in splits:
            # hard coded dataset_id of 0 for samples optimized away from a toxic output
            away_inputs = splits[0]
        if 1 in splits:
            # hard coded dataset_id of 1 for samples optimized towards a safety output
            toward_inputs = splits[1]
        if 2 in splits:
            # hard coded dataset_id of 2 for samples optimized towards a utility dataset output
            utility_inputs = splits[2]

        return toward_inputs, away_inputs, utility_inputs

    # TODO comment this code
    def get_away_perturbation_from_toward_perturbation(
        self, toward_inputs, adv_perturbation, adv_perturbation_mask
    ):
        # we can use the same perturbation as for the away loss (adversarial attack that triggers harmful target)
        # but we will calculate the loss for the safe target
        # TODO this function wil throw an error if the instruction of toward_inputs and away_inputs is tokenized differently!
        vocab_size = self.embed_weights.shape[0]
        embedding_size = self.embed_weights.shape[1]



        toward_target_ids = toward_inputs["labels"]
        toward_input_mask = toward_target_ids < 0
    
      
        toward_input_mask = (toward_input_mask * toward_inputs["attention_mask"]).to(bool)
      
        adv_perturbation_mask = adv_perturbation_mask.expand(-1, -1, embedding_size).to(bool)  # TODO FIX
       
        masked_perturbation = adv_perturbation[adv_perturbation_mask]

     

        inputs_toward = toward_inputs["input_ids"]
 
        one_hot = torch.zeros(
            (*inputs_toward.shape, vocab_size),
            dtype=self.embed_weights.dtype,
            device=self.embed_weights.device,
        )
      
        one_hot.scatter_(2, inputs_toward.unsqueeze(2), 1)
      
        toward_embeds = one_hot @ self.embed_weights

        flattened_inputs_toward = toward_inputs["input_ids"][toward_input_mask]
     

        one_hot = torch.zeros(
            (flattened_inputs_toward.shape[0], vocab_size),
            dtype=self.embed_weights.dtype,
            device=self.embed_weights.device,
        )


        one_hot.scatter_(1, flattened_inputs_toward.unsqueeze(1), 1)

        flattenend_embeds_toward = (one_hot @ self.embed_weights).flatten()
    
        flattened_perturbation = masked_perturbation + flattenend_embeds_toward
        perturbted_inputs_embeds_toward = toward_embeds
        perturbted_inputs_embeds_toward[toward_input_mask] = flattened_perturbation.view(-1, embedding_size)

        return perturbted_inputs_embeds_toward


class AdversarialDPOTrainer(AdversarialULTrainer):
    def __init__(
        self,
        adversarial_attack,
        embed_weights,
        scheduling=None,
        db_continue = 0,
        is_zero_a_valid_index=False,
        *args,
        **kwargs,
    ):
        super().__init__(adversarial_attack=adversarial_attack, embed_weights=embed_weights, is_zero_a_valid_index=is_zero_a_valid_index, *args, **kwargs)
        self.is_zero_a_valid_index=is_zero_a_valid_index
        self.db = db_continue
        self.scheduling  = scheduling
        self.ref_adapter_name = None
        self.is_peft_model = isinstance(self.model, PeftModel)

    def compute_loss(self, model, inputs, return_outputs=False):
        self.db = self.db +1       

        # inputs keys (['input_ids', 'attention_mask', 'labels', 'dataset_id'])
        toward_inputs, away_inputs, utility_inputs = self.split_inputs(inputs)

        away_loss, toward_loss, utility_loss, dpo_loss, attack_loss = (
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
        )
        away_text, utility_text = "", ""
        attack_losses, affirmative_responses = [], []

        # ======= away loss =======
        if away_inputs is not None:
            if self.scheduling is not None:
                temp_ratio = self.scheduling[self.db-1]
            else:
                temp_ratio = 1.0
            input_embeds, adv_perturbation, adv_perturbation_mask, attack_losses, affirmative_responses, is_gcg = (
                self.adversarial_attack.attack(
                    model,
                    away_inputs["input_ids"],
                    away_inputs["labels"],
                    away_inputs["attention_mask"],
                    ratio = temp_ratio
                )
            )
            if len(attack_losses) > 0:
                attack_loss = sum(attack_losses) / len(attack_losses)
            else:
                raise "No attack loss"
            perturbted_inputs_embeds_away = self.adversarial_attack.get_adv_embeddings(
                input_embeds, adv_perturbation, adv_perturbation_mask, is_gcg
            )
            outputs = model(
                inputs_embeds=perturbted_inputs_embeds_away,
                attention_mask=away_inputs["attention_mask"],
                labels=away_inputs["labels"],
            )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            away_logits = outputs[1]
            away_text = repr(
                model_utils.logits_to_text(
                    away_logits[0, (away_inputs["labels"][0] != -100).nonzero().squeeze() - 1].unsqueeze(0),
                    self.tokenizer,
                )
            )
            away_logps = self.get_batch_logps(
                away_logits,
                away_inputs["labels"],
                average_log_prob=self.dpo_loss_type == "ipo",
                is_encoder_decoder=self.is_encoder_decoder,
                label_pad_token_id=self.label_pad_token_id,
                upper_cutoff=0.0,
                lower_cutoff=self.away_cutoff,
            )
            away_loss = -outputs[0]
            

        # ======= toward loss =======
        if toward_inputs is not None:
            if is_gcg:
                for i in range(toward_inputs["input_ids"].shape[0]):
                    if self.is_zero_a_valid_index:
                        target_mask_away = away_inputs["labels"][i] >= 0
                    else:
                        target_mask_away = away_inputs["labels"][i] > 0
                
                    input_mask_away = (~target_mask_away * away_inputs["attention_mask"][i]).to(bool)
                    if self.is_zero_a_valid_index: 
                        target_mask_toward = toward_inputs["labels"][i] >= 0
                    else:
                        target_mask_toward = toward_inputs["labels"][i] > 0
                 
                    input_mask_toward = (~target_mask_toward * toward_inputs["attention_mask"][i]).to(bool)
                    temp = away_inputs["input_ids"][i,input_mask_away]

                    index = torch.nonzero(input_mask_toward).squeeze(-1)[-1].item()
                    toward_inputs["input_ids"][i,index+1-temp.shape[0]:index+1] = temp
                    toward_inputs["attention_mask"][i,index+1-temp.shape[0]:index+1] = 1
            perturbted_inputs_embeds_toward = self.get_away_perturbation_from_toward_perturbation(
                toward_inputs, adv_perturbation, adv_perturbation_mask
            )

            outputs = model(
                inputs_embeds=perturbted_inputs_embeds_toward,
                attention_mask=toward_inputs["attention_mask"],
                labels=toward_inputs["labels"],
            )
            toward_logits = outputs[1]
            toward_loss = outputs[0]
            toward_logps = self.get_batch_logps(
                toward_logits,
                toward_inputs["labels"],
                average_log_prob=self.dpo_loss_type == "ipo",
                is_encoder_decoder=self.is_encoder_decoder,
                label_pad_token_id=self.label_pad_token_id,
                upper_cutoff=-self.toward_cutoff,
                lower_cutoff=-100000,
            )

        # ======= utility loss =======
        if utility_inputs is not None:
            outputs = model(
                input_ids=utility_inputs["input_ids"],
                attention_mask=utility_inputs["attention_mask"],
                labels=utility_inputs["labels"],
            )
            utility_logits = outputs[1]
            utility_text = repr(
                model_utils.logits_to_text(
                    utility_logits[
                        0, (utility_inputs["labels"][0] != -100).nonzero().squeeze() - 1
                    ].unsqueeze(0),
                    self.tokenizer,
                )
            )
            utility_loss = outputs[0]

        if away_inputs is not None:
            dpo_loss = get_dpo_loss(
                self, toward_logps, away_logps, toward_inputs["logps"], away_inputs["logps"]
            )[0].mean()
        loss = self.dpo_weight * dpo_loss + self.utility_weight * utility_loss

        log(
            self,
            loss,
            away_loss,
            toward_loss,
            dpo_loss,
            utility_loss,
            attack_losses,
            attack_loss,
            affirmative_responses,
            away_text,
            utility_text,
        )



        return loss

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """
        with torch.no_grad():
            if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
                dataloader_params = {
                    "batch_size": self.args.per_device_train_batch_size,
                    "collate_fn": self.data_collator,
                    "num_workers": self.args.dataloader_num_workers,
                    "pin_memory": self.args.dataloader_pin_memory,
                    "shuffle": False,
                }

                # prepare dataloader
                data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

                # We compute logps for away toward and utility data - utility data is not needed for DPO
                valid_dataset_ids = data.get_dataset_ids()
                different_logps = {k: [] for k in valid_dataset_ids}

                for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                    dataset_ids = padded_batch["dataset_id"]
                    logp = self.compute_reference_log_probs(padded_batch)
                    for k in valid_dataset_ids:
                        mask = dataset_ids == k
                        different_logps[k].extend(logp[mask].cpu().tolist())

                # different_logps = {
                #     key: torch.cat(different_logps[key]).float().numpy() for key in valid_dataset_ids
                # }
                def assign_logps(ex, logps):
                    if ex["dataset_id"] == 2:
                        return {"logps": logps[2].pop(), **ex}
                    else:
                        safe_model = ex.pop("Safe_Model")
                        safe_model["logps"] = logps[1].pop()
                        return {"logps": logps[0].pop(), "Safe_Model": safe_model, **ex}

                ## DO NOT PARALLELISE ##
                self.train_dataset = self.train_dataset.map(
                    assign_logps, fn_kwargs={"logps": different_logps}
                )

                self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(
            self.model
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = (
            torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
        )

        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager():
            if self.dpo_reference_model is None:
                with self.null_ref_context():
                    logits = self.model(
                        padded_batch["input_ids"],
                        attention_mask=padded_batch["attention_mask"],
                        labels=padded_batch["labels"],
                        use_cache=False,
                    ).logits
                    logps = DPOTrainer.get_batch_logps(
                        logits,
                        padded_batch["labels"],
                        average_log_prob=self.dpo_loss_type == "ipo",
                        is_encoder_decoder=self.is_encoder_decoder,
                        label_pad_token_id=self.label_pad_token_id,
                    )
            else:
                logits = self.dpo_reference_model(
                    padded_batch["input_ids"],
                    attention_mask=padded_batch["attention_mask"],
                    labels=padded_batch["labels"],
                    use_cache=False,
                ).logits
                logps = DPOTrainer.get_batch_logps(
                    logits,
                    padded_batch["labels"],
                    average_log_prob=self.dpo_loss_type == "ipo",
                    is_encoder_decoder=self.is_encoder_decoder,
                    label_pad_token_id=self.label_pad_token_id,
                )

        return logps

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
        upper_cutoff: float = -10.0,
        lower_cutoff: float = 0.0,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        # Zero out the loss for tokens where log(P(label)) is less than the threshold
        below_threshold = per_token_logps < lower_cutoff
        per_token_logps[below_threshold] = 0

        # Zero out the loss for tokens where log(P(label)) is greater than the threshold
        above_threshold = per_token_logps > upper_cutoff
        per_token_logps[above_threshold] = 0

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)


def log_1_minus_p_loss(logits, labels, threshold=-5.0):
    # Compute the log(sum(exp(logits))) for each token position
    log_sum_exp_all = torch.logsumexp(logits, dim=-1)

    # Temporarily replace -100 labels with 0 for the gather operation
    gather_labels = labels.clone()
    gather_labels[labels == -100] = 0

    # Get the logits corresponding to the labels
    logits_for_labels = torch.gather(logits, -1, gather_labels.unsqueeze(-1)).squeeze(-1)

    # Calculate log(P(label))
    log_p = logits_for_labels - log_sum_exp_all

    # Create a mask for the labels, so we can zero out the logits of true labels
    mask = torch.zeros_like(logits).scatter_(-1, gather_labels.unsqueeze(-1), 1.0)

    # Zero out the logits of true labels
    mask_value = torch.finfo(logits.dtype).min  # Large negative value to approximate zero when exponentiated
    masked_logits = logits * (1 - mask) + mask * mask_value

    # Compute the log(sum(exp(logits excluding true label))) for each token position
    log_sum_exp_without_true_label = torch.logsumexp(masked_logits, dim=-1)

    # Compute log(1 - P(label)) for each token position
    log_1_minus_p = log_sum_exp_without_true_label - log_sum_exp_all

    # Set losses for -100 labels to 0 (ignored values)
    ignored_values = labels == -100
    log_1_minus_p[ignored_values] = 0

    # Zero out the loss for tokens where log(P(label)) is less than the threshold
    below_threshold = log_p < threshold
    log_1_minus_p[below_threshold] = 0

    # Compute the mean of the log(1 - P(label)) values, excluding the ignored ones
    loss = -log_1_minus_p.sum() / (~ignored_values).sum().float()

    return loss


def get_dpo_loss(
    trainer,
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    if trainer.reference_free:
        ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
    else:
        ref_logratios = reference_chosen_logps - reference_rejected_logps

    pi_logratios = pi_logratios.to(trainer.accelerator.device)
    ref_logratios = ref_logratios.to(trainer.accelerator.device)
    logits = pi_logratios - ref_logratios

    # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
    # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
    # calculates a conservative DPO loss.
    if trainer.dpo_loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(trainer.beta * logits) * (1 - trainer.label_smoothing)
            - F.logsigmoid(-trainer.beta * logits) * trainer.label_smoothing
        )
    elif trainer.dpo_loss_type == "hinge":
        losses = torch.relu(1 - trainer.beta * logits)
    elif trainer.dpo_loss_type == "ipo":
        # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
        losses = (logits - 1 / (2 * trainer.beta)) ** 2
    elif trainer.dpo_loss_type == "kto_pair":
        # eqn (7) of the HALOs paper
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
        losses = torch.cat(
            (
                1 - F.sigmoid(trainer.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(trainer.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        )
    else:
        raise ValueError(
            f"Unknown loss type: {trainer.dpo_loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
        )

    chosen_rewards = (
        trainer.beta
        * (
            policy_chosen_logps.to(trainer.accelerator.device)
            - reference_chosen_logps.to(trainer.accelerator.device)
        ).detach()
    )
    rejected_rewards = (
        trainer.beta
        * (
            policy_rejected_logps.to(trainer.accelerator.device)
            - reference_rejected_logps.to(trainer.accelerator.device)
        ).detach()
    )

    return losses, chosen_rewards, rejected_rewards


def get_writer_callback(trainer):
    for cb in trainer.callback_handler.callbacks:
        if type(cb) == TensorBoardCallback:
            return cb
    return None


def log_hparams(trainer, hparams, metrics):
    hparams = {k: str(hparams[k]) for k in hparams}
    cb = get_writer_callback(trainer)
    cb.tb_writer.add_hparams(hparams, metrics, run_name="hparams")


def prep_for_log(v):
    if isinstance(v, torch.Tensor):
        return v.item()
    return v


def log(
    trainer,
    loss,
    away_loss,
    toward_loss,
    dpo_loss,
    utility_loss,
    attack_losses,
    attack_loss,
    affirmative_responses,
    away_text,
    utility_text,
):
    logging.info(
        (
            f"Total loss {loss}; "
            f"Away loss {away_loss}; "
            f"Toward loss {toward_loss}; "
            f"DPO loss {dpo_loss}; "
            f"Utility loss {utility_loss}; "
            f"All attack losses {attack_losses}; "
            f"Affirmative responses {affirmative_responses}; "
            f"Away-Toward output {away_text}; "
            f"Utility output {utility_text}"
        )
    )

    metrics = {
        "global_step": trainer.state.global_step,
        "loss": prep_for_log(loss),
        "away_loss": prep_for_log(away_loss),
        "toward_loss": prep_for_log(toward_loss),
        "utility_loss": prep_for_log(utility_loss),
        "dpo_loss": prep_for_log(dpo_loss),
        "attack_loss": prep_for_log(attack_loss),
    }
    last_step = len(trainer.callback_handler.train_dataloader) * trainer.state.num_train_epochs == (
        trainer.state.global_step + 1
    )
    if last_step:
        log_hparams(trainer, trainer.hparams, metrics)
    trainer.log(metrics)
