def build_postprocess_prompt(competition_slug,
                             competition_overview,
                             data_description,
                             discussion_texts):
    return f"""
    You are a Kaggle competition assistant.

    Given the following data:

    Overview:
    {competition_overview}

    Data Description:
    {data_description}

    Top discussions and shared solutions from participants:
    {discussion_texts}

    Return a JSON object with the following fields:

    - "competition_slug": "{competition_slug}",
    - "overview_summary": a clean and concise summary of the competition overview (remove html, promotional or redundant phrases),
    - "data_description_clean": a cleaned version of the dataset description, keeping only relevant technical or domain information,
    - "feature_insights": a detailed description of the most important features or feature engineering blocks that helped participants achieve good results.
    - "modeling_strategies": a summary of the most common modeling strategies used by participants, including any specific algorithms or techniques that were particularly effective.

    Only return the JSON. Do not include explanations.
    """
    
def build_inference_prompt(new_comp_overview: str,
                           new_comp_data_desc: str,
                           contexts: list[str]) -> str:
    
    context_block = "\n---\n".join(contexts)
    
    prompt = f"""
    You are a Kaggle competition expert.

    Based on the following discussion excerpts from previous similar competitions:

    {context_block}

    Now, a new competition has the following overview:
    {new_comp_overview}

    And the following data description:
    {new_comp_data_desc}

    Please generate a list of feature ideas that are likely to work well for this new competition. For each feature, provide a short explanation of why it may be useful, referencing relevant techniques if needed.

    Return **only** a valid JSON array of dictionaries, where each dictionary contains:
    - "feature_name": a short name or label for the feature,
    - "description": a brief explanation or rationale for the feature.

    Do not include any text outside the JSON.
    """
    return prompt