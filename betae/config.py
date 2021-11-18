config = {
  "cuda": True, "do_train": True, "do_valid": False, "do_test": False,
  "data_path": "data/FB15k-betae",
  "negative_sample_size": 128,	# negative entities sampled per query
  "hidden_dim": 400,	# embedding dimension
  "gamma": 60.0,	# margin in the loss
  "batch_size": 512,	# batch size of queries
  "test_batch_size": 1,	# valid/test batch size
  "learning_rate": 0.0001,
  "cpu_num": 1,	# used to speed up torch.dataloader
  "save_path": "./saves",	# no need to set manually, will configure automatically
  "max_steps": 1000, # 450001,  # maximum iterations to train
  "warm_up_steps": None,	# no need to set manually, will configure automatically

  "save_checkpoint_steps": 50000,  # save checkpoints every xx steps
  "valid_steps":  500, #15000,	# evaluate validation queries every xx steps
  "log_steps": 100,	# train log every xx steps
  "test_log_steps": 100, # 1000,	# valid/test log every xx steps

  "nentity": 0,  # DO NOT MANUALLY SET
  "nrelation": 0,	# DO NOT MANUALLY SET

  "geo": "beta",	# the reasoning model, vec for GQE, box for Query2box, beta for BetaE
  "print_on_screen": True,	# print_on_screen

  "tasks": "1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up",	# tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task 
  "seed": 0,	# random seed
  "beta_mode": "(1600,2)",	# (hidden_dim,num_layer for BetaE relational projection)
  # "box_mode": "(none,0.02)",	# (offset activation,center_reg for Query2box, center_reg balances the in_box dist and out_box dist)
  "prefix": "logs",	# prefix of the log path
  "checkpoint_path": None,	# path for loading the checkpoints
  "evaluate_union": "DNF" # the way to evaluate union queries, transform it to disjunctive normal form (DNF or use the De Morgan\s laws (DM))
}