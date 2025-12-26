import argparse
import ast
import os
import numpy as np
import torch
import ujson as json
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation_revised import to_official, official_evaluate
# import wandb
from losses import get_label, get_at_loss, get_at_loss_by_MT, get_label_by_MT


def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        best_dev_output = None
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        progress_bar = tqdm(total=total_steps, desc="Training Progress")
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            print('epoch: {}'.format(epoch))
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
                model.train()
                inputs = {'input_ids': batch[0].to('cuda'),
                          'attention_mask': batch[1].to('cuda'),
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
                labels = batch[2].to('cuda')
                with torch.cuda.amp.autocast():
                    logits = model(**inputs)
                    if args.loss_type == 'ATL':
                        loss = get_at_loss(logits, labels)
                    elif args.loss_type == 'ATL_MT':
                        loss = get_at_loss_by_MT(logits, labels, Na_idxs=ast.literal_eval(args.MT_Na),MT_lambda=args.MT_lambda)
                    loss = loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                if step % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0
                                                               and num_steps % args.evaluation_steps == 0
                                                               and step % args.gradient_accumulation_steps == 0):
                    lr = optimizer.state_dict()['param_groups'][0]['lr']
                    print("epoch:{}, lr:{}".format(epoch, lr))
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    print(dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        best_dev_output = dev_output
                        print('best f1: {}'.format(best_score))
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)
                progress_bar.update(1)
        return best_dev_output

    scaler = torch.cuda.amp.GradScaler()
    new_layer = ["extractor", "bilinear", "projector", "classifier"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_steps = 0
    model.zero_grad()
    best_dev_output = finetune(train_features, optimizer, args.num_train_epochs, num_steps)
    return best_dev_output


def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    logitss = []
    model.eval()
    for batch in dataloader:
        inputs = {'input_ids': batch[0].to('cuda'),
                  'attention_mask': batch[1].to('cuda'),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = model(**inputs)
            if args.MT:
                pred = get_label_by_MT(args, logits, num_labels=args.num_labels, Na_idxs=ast.literal_eval(args.MT_Na))
            else:
                pred = get_label(args, logits, num_labels=args.num_labels)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            logitss.append(logits.cpu().numpy())

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    logitss = np.concatenate(logitss, axis=0).astype(np.float32)
    ans = to_official(preds, logitss, features, args.num_class, args.rel2id_file_path)
    best_f1, best_f1_ign, p, r = 0, 0, 0, 0
    if len(ans) > 0:
        if tag == 'dev':
            best_f1, _, best_f1_ign, _, p, r = official_evaluate(ans, args.data_dir, args.train_file, args.dev_file)
        else:
            best_f1, _, best_f1_ign, _, p, r = official_evaluate(ans, args.data_dir, args.train_file, args.test_file)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
        tag + "_p": p * 100,
        tag + "_r": r * 100
    }
    return best_f1, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_name", default="", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument("--neg_lambda", type=float, default=1,
                        help="Loss coefficient of negative samples.")
    parser.add_argument("--gamma", type=float, default=1,
                        help="Focal loss gamma.")
    parser.add_argument("--proj_name", default="Re-DocRED", type=str,
                        help="wandb project name")
    parser.add_argument("--run_name", default="", type=str,
                        help="wandb run name")
    parser.add_argument("--loss_type", default="ATL", type=str,
                        help="loss type: ATL/balance_softmax/AFL/SAT/MeanSAT/HingeABL")
    parser.add_argument("--neg_sample_rate", type=float, default=1,
                        help="Negative sampling rate.")
    parser.add_argument("--margin", type=float, default=5,
                        help="hinge margin.")
    parser.add_argument('--nseed', nargs='+', type=int, default=[])
    parser.add_argument('--disable_log', action='store_true')
    parser.add_argument('--pos_only', action='store_true')
    parser.add_argument("--rel2id_file_path", default='meta/rel2id.json', type=str)
    parser.add_argument("--MT",action='store_true')
    parser.add_argument("--cuda_device", type=int, default=0,
                        help="0/1/2/3")
    parser.add_argument("--MT_Na",default="", type=str)
    parser.add_argument("--MT_lambda",default=1,type=float)
    args = parser.parse_args()

    torch.cuda.set_device(args.cuda_device)
    args.n_gpu = torch.cuda.device_count()

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)


    read = read_docred


    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    print(args)
    model = DocREModel(config, model, num_labels=args.num_labels)
    model.to(args.cuda_device)
    if args.load_path == "":  # Training
        train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length,
                              rel2id_file_path=args.rel2id_file_path)
        dev_features = read(dev_file, tokenizer, rel2id_file_path=args.rel2id_file_path,
                            max_seq_length=args.max_seq_length)
        test_features = read(test_file, tokenizer, rel2id_file_path=args.rel2id_file_path,
                             max_seq_length=args.max_seq_length)
        result_item = dict()
        result_item['seed'] = args.seed
        args.save_path = 'checkpoint/' + args.save_name + '.pt'
        best_dev_output = train(args, model, train_features, dev_features, test_features)
        with open('result/' + args.save_name + '_dev.json', 'w') as f:
            json.dump(best_dev_output, f)
        model.load_state_dict(torch.load(args.save_path))
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        result_item['result'] = test_output
        print('seed:{}, result:{}'.format(args.seed, test_output))
        with open('result/' + args.save_name + '_test.json', 'w') as f:
            json.dump(result_item, f)
    else:  # Testing
        dev_features = read(dev_file, tokenizer, rel2id_file_path=args.rel2id_file_path,
                            max_seq_length=args.max_seq_length)
        test_features = read(test_file, tokenizer, rel2id_file_path=args.rel2id_file_path,
                             max_seq_length=args.max_seq_length)
        model.load_state_dict(torch.load(args.load_path))
        result_item = dict()
        result_item['seed'] = args.seed
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        print(dev_output)
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        print(test_output)
        print('seed:{}, result:{}'.format(args.seed, test_output))
        result_item['result'] = test_output
        with open('result/' + args.save_name + '_test.json', 'w') as f:
            json.dump(result_item, f)

if __name__ == "__main__":
    main()
