import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import ast
from scipy.special import softmax

# Constants
models = ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
model_names = ["LLaMA3.2-1B", "LLaMA3.2-3B", "Qwen2.5-1.5B", "Qwen2.5-3B"]
max_new_tokens = 250
system_prompt = "You are an intelligent chatbot answering factual questions correctly based on your own knowledge. A paragraph is provided as well that may or may not be helpful information. Discern if the paragraph's contents help you answer the question or not accordingly. Just return the answer, do not explain."

def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  
        device_map="auto",
        attn_implementation="eager"
    )
    return model, tokenizer

def generate_prompt(context, question):
    messages = []

    messages.append({ "role": "system", "content": system_prompt })
    messages.append({ "role": "user", "content": f"Context: {context}" })
    messages.append({ "role": "user", "content": f"Question: {question}" })

    return messages

def generate(model, tokenizer, max_new_tokens, prompt):
    inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(inputs=inputs, return_dict_in_generate=True, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, output_attentions=True)
        output_sequence, output_attention = outputs.sequences, outputs.attentions
        answer = output_sequence[:, inputs.shape[-1]:][0, :]
    
    return answer, output_sequence, output_attention

def save_to_file(attentions, answer, model_name, index):
    to_save = {
        'attention_map': attentions,
        'answer': answer
    }

    torch.save(to_save, f"./{model_name}/entry_{index}.pt")

def prompt_model(model_name, model, tokenizer, max_new_tokens, context, question, index):
    prompt = generate_prompt(context, question)
    answer, output_sequence, output_attention = generate(model, tokenizer, max_new_tokens, prompt)
    save_to_file(output_attention, answer, model_name, index)

def to_token_weights(tokenizer, context, question, context_weights, question_weights):
    text = generate_prompt(context, question)
    index = 0
    tokenized_c_weights = []
    tokenized_q_weights = []
    context_words = context.split()
    question_words = question.split()
    tokenized_text = tokenizer.apply_chat_template(text, return_tensors="pt")
    tokens = tokenized_text[0].tolist()
    
    formatted_text = tokenizer.decode(tokenized_text[0], skip_special_tokens=True)

    context_start = formatted_text.find(context)
    question_start = formatted_text.find(question)
    
    # Get context
    str = ""
    i = 0
    while len(str) < context_start:
        t = tokenizer.decode(tokens[i], skip_special_tokens=True)

        if len(str) + len(t) >= context_start:
            break
        
        str += t 
        i += 1

    context_index = i
    token_buffer = ""
    word_buffer = context_words[0]
    while len(word_buffer) <= len(context):
        while len(token_buffer) - 1 < len(word_buffer):
            t = tokenizer.decode(tokens[i], skip_special_tokens=True)
            token_buffer += t
            str += t
            i += 1
            tokenized_c_weights.append(context_weights[index])

        index += 1

        if index >= len(context_words):
            break
            
        word_buffer += " "
        word_buffer += context_words[index]

    print(tokenizer.decode(tokens[i], skip_special_tokens=True))
    
    # Get questions
    while len(str) < question_start:
        t = tokenizer.decode(tokens[i], skip_special_tokens=True)

        if len(str) + len(t) >= question_start:
            break
        
        str += t 
        i += 1

    question_index = i
    token_buffer = ""
    word_buffer = question_words[0]
    index = 0
    while len(word_buffer) <= len(question):
        while len(token_buffer) - 1 < len(word_buffer):
            t = tokenizer.decode(tokens[i], skip_special_tokens=True)
            token_buffer += t
            str += t
            i += 1
            tokenized_q_weights.append(question_weights[index])

        index += 1

        if index >= len(question_words):
            break
        
        word_buffer += " "
        word_buffer += question_words[index]
    
    return context_index, tokenized_c_weights, question_index, tokenized_q_weights

def run_data_collection():
    dataset = pd.read_csv("reading_comprehension_attention_dataset.csv")

    for i in range(len(models)):
        model_id = models[i]
        model_name = model_names[i]
        model, tokenizer = load_model(model_id)
        
        print(f"Starting {model_name} runs")
        
        for index, row in dataset.iterrows():
            context = row['context']
            question = row['question']
    
            print(f"Prompting for index {index}")
            prompt_model(model_name, model, tokenizer, max_new_tokens, context, question, index)

def convert_data():
    dataset = pd.read_csv("reading_comprehension_attention_dataset.csv")
    context_weight = dataset['context_weight'].apply(ast.literal_eval)
    question_weight = dataset['question_weight'].apply(ast.literal_eval)
    data = dict()
    
    for i in range(len(models)):
        model_id = models[i]
        model_name = model_names[i]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        data[f'{model_name}_context_index'] = []
        data[f'{model_name}_question_index'] = []
        data[f'{model_name}_context_weights'] = []
        data[f'{model_name}_question_weights'] = []
    
        print(f"Starting {model_name} runs")
        
        for index, row in dataset.iterrows():
            context = row['context']
            question = row['question']
            context_weights = context_weight[index]
            question_weights = question_weight[index]
    
            print(f"Converting for index {index}")
            context_index, tokenized_c_weights, question_index, tokenized_q_weights = to_token_weights(tokenizer, context, question, context_weights, question_weights)
            data[f'{model_name}_context_index'].append(context_index)
            data[f'{model_name}_question_index'].append(question_index)
            data[f'{model_name}_context_weights'].append(tokenized_c_weights)
            data[f'{model_name}_question_weights'].append(tokenized_q_weights)

        dataset[f'{model_name}_context_index'] = data[f'{model_name}_context_index']
        dataset[f'{model_name}_question_index'] = data[f'{model_name}_question_index']
        dataset[f'{model_name}_context_weights'] = data[f'{model_name}_context_weights']
        dataset[f'{model_name}_question_weights'] = data[f'{model_name}_question_weights']

    dataset.to_csv("analysis_dataset.csv", index=None)
            
def validate_data():
    dataset = pd.read_csv("analysis_dataset.csv")
    context_weight = dataset['context_weight'].apply(ast.literal_eval)
    question_weight = dataset['question_weight'].apply(ast.literal_eval)

    for i in range(0, 2):
        model_id = models[i]
        model_name = model_names[i]
        model, tokenizer = load_model(model_id)

        context = dataset['context'][0]
        question = dataset['question'][0]
        cweights = (dataset[f'{model_name}_context_weights'].apply(ast.literal_eval))[0]
        qweights = (dataset[f'{model_name}_question_weights'].apply(ast.literal_eval))[0]
        ec = tokenizer.encode(context)
        eq = tokenizer.encode(question)

        for j in range(1, len(ec)):
            print(f"{tokenizer.decode(ec[j])}: {cweights[j - 1]}")

        for j in range(1, len(eq)):
            print(f"{tokenizer.decode(eq[j])}: {qweights[j - 1]}")

        print("\n\n")

    for i in range(2, len(models)):
        model_id = models[i]
        model_name = model_names[i]
        model, tokenizer = load_model(model_id)

        context = dataset['context'][0]
        question = dataset['question'][0]
        cweights = (dataset[f'{model_name}_context_weights'].apply(ast.literal_eval))[0]
        qweights = (dataset[f'{model_name}_question_weights'].apply(ast.literal_eval))[0]
        ec = tokenizer.encode(context)
        eq = tokenizer.encode(question)

        for j in range(len(ec)):
            print(f"{tokenizer.decode(ec[j])}: {cweights[j]}")

        for j in range(len(eq)):
            print(f"{tokenizer.decode(eq[j])}: {qweights[j]}")

        print("\n\n")

def test():
    dataset = pd.read_csv("analysis_dataset.csv")
    context_weight = dataset['context_weight'].apply(ast.literal_eval)
    question_weight = dataset['question_weight'].apply(ast.literal_eval)

    for i in range(0, 2):
        model_id = models[i]
        model_name = model_names[i]
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        print(f"Starting {model_name} runs")
        
        ci = dataset[f'{model_name}_context_index']
        qi = dataset[f'{model_name}_question_index']
        cw = dataset[f'{model_name}_context_weights'].apply(ast.literal_eval)
        qw = dataset[f'{model_name}_question_weights'].apply(ast.literal_eval)
    
        for x in range(len(ci)):
            print(f"Testing for index {x}")
            
            attention_map = torch.load(f"./{model_name}/entry_{x}.pt", map_location=torch.device("cpu"))
            attention_map = attention_map['attention_map']
            
            context = dataset['context']
            question = dataset['question']
    
            prompt = generate_prompt(context[x], question[x])
            tokens = tokenizer.apply_chat_template(prompt, return_tensors="pt")[0].tolist()
            if qi[x] + len(qw[x]) - 1 != len(tokens) - 2:
                
                print(attention_map[0][0].shape[-1])
                print(len(tokens))
                print()
                print(ci[x] + len(cw[x]) - 1)
                print(qi[x] + len(qw[x]) - 1)
                print()
                print(tokenizer.decode(tokens[ci[x]: ci[x] + len(cw[x])], skip_special_tokens=True))
                print(tokenizer.decode(tokens[qi[x]: qi[x] + len(qw[x])], skip_special_tokens=True))
                print()
                print("------------------------------------------------------")
            
                print("\n\n\n")

    for i in range(2, len(models)):
        model_id = models[i]
        model_name = model_names[i]
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        print(f"Starting {model_name} runs")
        
        ci = dataset[f'{model_name}_context_index']
        qi = dataset[f'{model_name}_question_index']
        cw = dataset[f'{model_name}_context_weights'].apply(ast.literal_eval)
        qw = dataset[f'{model_name}_question_weights'].apply(ast.literal_eval)
    
        for x in range(len(ci)):
            print(f"Testing for index {x}")
            
            attention_map = torch.load(f"./{model_name}/entry_{x}.pt", map_location=torch.device("cpu"))
            attention_map = attention_map['attention_map']
            
            context = dataset['context']
            question = dataset['question']
    
            prompt = generate_prompt(context[x], question[x])
            tokens = tokenizer.apply_chat_template(prompt, return_tensors="pt")[0].tolist()
            if qi[x] + len(qw[x]) - 1 != len(tokens) - 3:
                
                print(attention_map[0][0].shape[-1])
                print(len(tokens))
                print()
                print(ci[x] + len(cw[x]) - 1)
                print(qi[x] + len(qw[x]) - 1)
                print()
                print(tokenizer.decode(tokens[ci[x]: ci[x] + len(cw[x])], skip_special_tokens=True))
                print(tokenizer.decode(tokens[qi[x]: qi[x] + len(qw[x])], skip_special_tokens=True))
                print()
                print("------------------------------------------------------")
            
                print("\n\n\n")

def calc_attentions():
    dataset = pd.read_csv("analysis_dataset.csv")
    results = {}

    for i in range(len(models)):
        model_id = models[i]
        model_name = model_names[i]
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        print(f"Starting {model_name} runs")

        ci = dataset[f'{model_name}_context_index']
        qi = dataset[f'{model_name}_question_index']
        cw = dataset[f'{model_name}_context_weights'].apply(ast.literal_eval)
        qw = dataset[f'{model_name}_question_weights'].apply(ast.literal_eval)

        results[f'{model_name}_context_ave'] = []
        results[f'{model_name}_question_ave'] = []
        results[f'{model_name}_answer'] = []
        
        for x in range(len(ci)):
            print(f"Calculating for index {x}")

            output = torch.load(f"./{model_name}/entry_{x}.pt", map_location=torch.device("cpu"))
            attentions = output['attention_map']
            ans = tokenizer.decode(output['answer'], skip_special_tokens=True)

            context = dataset['context']
            question = dataset['question']
    
            prompt = generate_prompt(context[x], question[x])
            tokens = tokenizer.apply_chat_template(prompt, return_tensors="pt")[0].tolist()

            context_len = len(cw[x])
            question_len = len(qw[x])
            new_tokens = len(attentions)
            layers = len(attentions[0])
            
            context_attns = torch.zeros(new_tokens, layers, context_len)
            question_attns = torch.zeros(new_tokens, layers, question_len)
            
            for t in range(new_tokens):
                for l in range(layers):
                    context_attn = attentions[t][-1][0, :, -1, ci[x]:ci[x]+len(cw[x])].mean(0)
                    question_attn = attentions[t][-1][0, :, -1, qi[x]:qi[x]+len(qw[x])].mean(0)
                    context_attns[t, l, :] = context_attn
                    question_attns[t, l, :] = question_attn

            #Get average attention weights for the context and question
            context_ave = torch.mean(context_attns, dim=(0, 1)).tolist()
            question_ave = torch.mean(question_attns, dim=(0, 1)).tolist()

            results[f'{model_name}_answer'].append(ans)
            results[f'{model_name}_context_ave'].append(context_ave)
            results[f'{model_name}_question_ave'].append(question_ave)

    df = pd.DataFrame(results, index=None)
    df.to_csv('results2.csv', index=None)

if __name__ == "__main__":
    calc_attentions()