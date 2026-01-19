import os
import re
import json
import argparse
from tqdm import tqdm
from typing import List, Any
from prompter.system_prompt import SystemPrompter
from utils.langchain_utils import universal_model_loader


def response_generation(
    llm,
    model_name: str,
    message_list,
    temperature: float,
    top_p: float,
    max_tokens: int
) -> str:
    '''
        Generate response from the model based on the message list.
    '''
    
    # getting the inference using llm
    if 'gpt-5' in model_name or 'o4' in model_name:
        response_batch = llm.batch(
            message_list,
            max_tokens=max_tokens,
        )
    else:
        response_batch = llm.batch(
            message_list,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

    return [response.content for response in response_batch]


def document_type_generation(
    llm,
    model_name: str,
    universal_context_text: str,
    num_doc_types: int,
    temperature: float,
    top_p: float,
    max_tokens: int
) -> List[str]:
    '''
        Generate document types based on the universal context
    '''

    # first sampling the instruction
    doc_type_instruction = SystemPrompter.create_system_prompt('doc_type_generation')

    # creating the message
    message = [{
        'role': 'user',
        'content': f"{universal_context_text}\n\n{doc_type_instruction}"
    }]
    message_list = [message for _ in range(num_doc_types)]

    # generating the document types
    response_content = response_generation(
        llm=llm,
        model_name=model_name,
        message_list=message_list,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

    # processing the document types
    document_types = set()
    for content in response_content:
        lines = content.split('\n')
        for line in lines:
            if '- ' in line:
                doc_type = line.split('- ')[1].strip()
                document_types.add(doc_type)
    
    return list(document_types)


def idea_generation(
    llm,
    model_name: str,
    universal_context_text: str,
    document_types: List[str],
    num_ideas: int,
    temperature: float,
    top_p: float,
    max_tokens: int
) -> List[Any]:
    '''
        Generate ideas for each document type based on the universal context
    '''

    # storing all the ideas
    all_ideas = []

    # iterating over each of the document types
    for document_type in tqdm(document_types):

        # creating the instruction using the system prompt
        idea_instruction = SystemPrompter.create_system_prompt(
            'idea_generation',
            document_type=document_type
        )

        # creating the message
        message = [{
            'role': 'user',
            'content': f"{universal_context_text}\n\n{idea_instruction}"
        }]
        message_list = [message for _ in range(num_ideas)]

        # generating the ideas
        response_content = response_generation(
            llm=llm,
            model_name=model_name,
            message_list=message_list,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

        # processing the ideas
        current_ideas = {
            'document_type': document_type,
            'ideas': []
        }
        for content in response_content:

            # using regex to extract ideas present inside <idea>...</idea>
            ideas = re.findall(r'<idea>(.*?)</idea>', content, re.DOTALL)
            for idea in ideas:
                current_ideas['ideas'].append(idea.strip())

        all_ideas.append(current_ideas)

    return all_ideas


def document_content_generation(
    llm,
    model_name: str,
    universal_context_text: str,
    document_ideas: List[Any],
    num_generations: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    output_folder: str
) -> List[Any]:
    '''
        Generate document contents based on the universal context and document ideas
    '''

    # global index to save the document
    doc_index = 1

    # iterating over each of the document ideas
    for idea_entry in tqdm(document_ideas, desc='Document Types'):
        document_type = idea_entry['document_type']
        ideas = idea_entry['ideas']

        for idea in tqdm(ideas, desc='Ideas'):
            
            # creating the instruction using the system prompt
            content_instruction = SystemPrompter.create_system_prompt(
                'doc_generation',
                document_type=document_type,
                idea=idea
            )

            # creating the message
            message = [{
                'role': 'user',
                'content': f"{universal_context_text}\n\n{content_instruction}"
            }]
            message_list = [message for _ in range(num_generations)]

            # generating the document content
            response_content = response_generation(
                llm=llm,
                model_name=model_name,
                message_list=message_list,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )

            # saving the document content
            for content in response_content:
                doc_filename = os.path.join(
                    output_folder,
                    f'document_{doc_index}.txt'
                )

                # extracting the content inside <scratchpad>...</scratchpad> and <content>...</content>
                scratchpad = '\n'.join(re.findall(r'<scratchpad>(.*?)</scratchpad>', content, re.DOTALL))
                content = '\n'.join(re.findall(r'<content>(.*?)</content>', content, re.DOTALL))

                # checking if content is non-empty and UNSUITABLE is not present in scratchpad
                if content.strip() and 'UNSUITABLE' not in scratchpad:
                    with open(doc_filename, 'w') as f:
                        f.write(content.strip())
                    doc_index += 1

    

def pipeline_document_generation(
    model_name: str,
    tensor_parallel_size: int,
    organization: str,
    universal_context: str,
    num_doc_types: int,
    num_idea_per_type: int,
    num_generations: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    few_shot_num: int,
    output_folder: str
) -> None:
    '''
        Pipeline to generate documents based on universal context.
    '''

    # Load the model
    llm, _ = universal_model_loader(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        seed=42
    )

    # loading the universal context through the system prompter
    universal_context_text = SystemPrompter.create_system_prompt(
        system_prompt_name=universal_context,
        company=organization
    )

    # making the save folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    # first generating document types
    if not os.path.exists(os.path.join(output_folder, 'document_types.json')):
        document_types_list = document_type_generation(
            llm=llm,
            model_name=model_name,
            universal_context_text=universal_context_text,
            num_doc_types=num_doc_types,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        with open(os.path.join(output_folder, 'document_types.json'), 'w') as f:
            json.dump(document_types_list, f, indent=4)
    else:
        with open(os.path.join(output_folder, 'document_types.json'), 'r') as f:
            document_types_list = json.load(f)

    # then generating ideas for each document type
    if not os.path.exists(os.path.join(output_folder, 'document_ideas.json')):
        document_ideas = idea_generation(
            llm=llm,
            model_name=model_name,
            universal_context_text=universal_context_text,
            document_types=document_types_list,
            num_ideas=num_idea_per_type,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        with open(os.path.join(output_folder, 'document_ideas.json'), 'w') as f:
            json.dump(document_ideas, f, indent=4)
    else:
        with open(os.path.join(output_folder, 'document_ideas.json'), 'r') as f:
            document_ideas = json.load(f)

    # generating ideas for each document type and idea configuration
    os.makedirs(os.path.join(output_folder, 'documents'), exist_ok=True)
    document_content_generation(
        llm=llm,
        model_name=model_name,
        universal_context_text=universal_context_text,
        document_ideas=document_ideas,
        num_generations=num_generations,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        output_folder=os.path.join(output_folder, 'documents')
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Document Generation Pipeline")
    parser.add_argument('--model_name', type=str, default='openai:gpt-4o')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--organization', type=str, default='AllenAI')
    parser.add_argument('--universal_context', type=str, default='mft_universal_context')
    parser.add_argument('--num_doc_types', type=int, default=5)
    parser.add_argument('--num_idea_per_type', type=int, default=2)
    parser.add_argument('--num_generations', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--few_shot_num', type=int, default=5, help='Number of few-shot examples to include.')
    parser.add_argument('--output_folder', type=str, default='training_output/mft_sa')
    args = parser.parse_args()

    pipeline_document_generation(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        organization=args.organization,
        universal_context=args.universal_context,
        num_doc_types=args.num_doc_types,
        num_idea_per_type=args.num_idea_per_type,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        few_shot_num=args.few_shot_num,
        output_folder=args.output_folder
    )

