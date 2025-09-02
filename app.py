import os
import sys
import argparse
import google.generativeai as genai
import json
import numpy as np
from openai import OpenAI


def get_system_prompt() -> str:
    """
    Returns the static system prompt that defines the AI's persona and primary objective.
    """
    return """
You are an expert IBM Maximo and Senior QA Engineer with 30 years of experience creating formal test case documentation for enterprise software applications at leading companies like IBM and Google.
Your primary objective is to create comprehensive, well-structured test cases that clearly outline verification steps for specific functionality, ensuring the application behaves as expected under various conditions.
Your task is to convert a user-provided scenario into a detailed, formal test case formatted in Markdown.
"""

def build_user_prompt(scenario: str, custom_context: str, example_section: str) -> str:
    """
    Constructs the user-facing part of the prompt, including context, instructions, and the specific scenario.
    """
    context_section = ""
    if custom_context:
        context_section = f"""
**Additional Custom Context Provided by User:**
The following information is from a user-provided document. Use it as a primary reference to understand specific processes, naming conventions, or data mentioned in the scenario.
---
{custom_context}
---
"""
    return f"""
First, review any additional custom context provided below, then proceed with the instructions.
{context_section}

**Instructions:**
1.  Analyze the user's scenario.
2.  Generate a comprehensive test case with the following sections:
    - **Test Case ID:** A unique identifier (e.g., TC-MAX-001).
    - **Title:** A concise and descriptive title based on the scenario.
    - **Objective:** A brief summary of what this test case aims to verify.
    - **Prerequisites:** A list of all necessary preconditions, such as:
        - User Roles/Permissions (e.g., Maintenance Supervisor, Storeroom Clerk).
        - Required Data (e.g., An approved Work Order with status 'APPR', a specific item in the storeroom).
        - System State (e.g., User is logged into Maximo).
    - **Test Steps:** A table with three columns: 'Actions', 'Expected Result', and 'Actual Result'. IMPORTANT: Do not use bold markdown (`**`) for UI elements like field names or buttons within the table cells.
    - **Test Data:** A section listing any specific data used in the test, like Work Order numbers, Asset numbers, or User IDs.
{example_section}

---
**User's Scenario to process:**
"{scenario}"
"""

def generate_maximo_test_case(scenario: str, api_keys: dict, custom_context: str, example_section: str, model_name: str) -> str:
    """
    Uses the Gemini API to generate a Maximo test case from a scenario.

    Args:
        scenario: The user-provided scenario string.
        api_keys: A dictionary containing API keys for different services.
        custom_context: Optional string containing user-specific documentation.
        example_section: The formatted example to be included in the prompt.
        model_name: The name of the AI model to use.

    Returns:
        The generated test case in Markdown format.
    """
    try:
        print(f"--> Generating test case with model: '{model_name}'")

        if "gemini" in model_name:
            # For Gemini, we combine the system and user prompts into a single prompt.
            full_prompt = get_system_prompt() + "\n\n" + build_user_prompt(scenario, custom_context, example_section)            
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(full_prompt)
            return response.text.strip().strip('```markdown').strip('```').strip()
        
        elif "gpt" in model_name:
            try:
                import openai
            except ImportError:
                raise ImportError("The 'openai' library is required to use GPT models. Please install it using: pip install openai")
            
            # For OpenAI, it's best practice to use a separate "system" message for the persona.
            system_prompt = get_system_prompt()
            user_prompt = build_user_prompt(scenario, custom_context, example_section)

            client = openai.OpenAI(api_key=api_keys.get('openai'))
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content.strip().strip('```markdown').strip('```').strip()
        
        else:
            raise ValueError(f"Unsupported or unknown model name: {model_name}")

    except Exception as e:
        print(f"An error occurred while communicating with the AI model '{model_name}': {e}", file=sys.stderr)
        raise # Re-raise the exception to be caught by the Flask app

def save_steps_to_excel(markdown_text: str, excel_filename: str):
    """
    Parses a Markdown string to find a test steps table and saves it to an Excel file.
    """
    try:
        import pandas as pd
    except ImportError:
        print("\nWarning: `pandas` and `openpyxl` are required to save to Excel.", file=sys.stderr)
        print("Please install them using: pip install pandas openpyxl", file=sys.stderr)
        return

    lines = markdown_text.split('\n')
    table_data = []
    
    # Simple state machine to parse the table
    # State 0: Looking for header
    # State 1: Found header, looking for separator
    # State 2: Found separator, reading data rows
    state = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            if state == 2: # Empty line after table data means table ended
                break
            continue

        if state == 0 and "Actions" in line and "Expected Result" in line and "Actual Result" in line and line.startswith('|'):
            state = 1
        elif state == 1 and line.startswith('|--'):
            state = 2
        elif state == 2 and line.startswith('|'):
            parts = [part.strip() for part in line.split('|')]
            if len(parts) >= 4:
                table_data.append({'Actions': parts[1], 'Expected Result': parts[2], 'Actual Result': parts[3]})
        elif state == 2:
            # We were reading data, but this line doesn't fit the pattern. Table must be over.
            break

    if not table_data:
        print("\nWarning: Could not find or parse test steps table in the generated Markdown. Excel file not created.")
        return

    try:
        df = pd.DataFrame(table_data)
        df.to_excel(excel_filename, index=False, engine='openpyxl')
        print(f"Successfully extracted test steps and saved to '{os.path.abspath(excel_filename)}'")
    except Exception as e:
        print(f"\nError saving test steps to Excel file '{excel_filename}': {e}", file=sys.stderr)

def modify_test_steps(steps_table_md: str, modification_instruction: str, api_keys: dict, model_name: str) -> str:
    """
    Takes a markdown table of test steps and a modification instruction,
    and returns the updated markdown table using the Gemini API.
    """
    prompt = f"""
You are an expert test case editor. Your task is to modify the provided Markdown table of test steps based on the user's instruction.

**Rules:**
1.  Return ONLY the complete, updated Markdown table.
2.  Do not include any explanations, notes, or text outside of the table.
3.  Ensure the output is valid Markdown. Do not use HTML tags like `<br>`.
4.  If a step is removed, renumber the subsequent steps sequentially starting from 1.
5.  The table must have three columns: "Actions", "Expected Result", "Actual Result".

**Original Test Steps Table:**
{steps_table_md}

**User's Instruction:**
"{modification_instruction}"

**Updated Markdown Table:**
"""
    try:
        print(f"--> Modifying test steps with model: '{model_name}'")
        if "gemini" in model_name:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text.strip().strip('```markdown').strip('```').strip()

        elif "gpt" in model_name:
            try:
                import openai
            except ImportError: 
                raise ImportError("The 'openai' library is required to use GPT models. Please install it using: pip install openai")
            
            client = openai.OpenAI(api_key=api_keys.get('openai'))
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip().strip('```markdown').strip('```').strip()
        else:
            raise ValueError(f"Unsupported or unknown model name: {model_name}")
    except Exception as e:
        print(f"An error occurred during test step modification: {e}", file=sys.stderr)
        raise

def classify_user_intent(user_input: str, api_keys: dict) -> str:
    """
    Uses a Gemini model to classify the user's intent as either 'GENERATE' or 'MODIFY'.
    """
    # We use a very fast model for this classification task.
    model_name = 'gemini-1.5-flash-latest'
    
    prompt = f"""
Analyze the user's request below. Is the user asking to generate a completely new test case from a scenario, or are they asking to modify an existing one?
Respond with only a single word: either "GENERATE" or "MODIFY".

Examples:
- Request: "create a test case for asset creation" -> GENERATE
- Request: "a user creates a work order and approves it" -> GENERATE
- Request: "add a new step to check the log file" -> MODIFY
- Request: "remove step 3" -> MODIFY
- Request: "change the objective to verify the email" -> MODIFY
- Request: "make step 2 more detailed" -> MODIFY

User's Request: "{user_input}"
"""
    try:
        print(f"--> Classifying intent for input: '{user_input}'")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        # Clean up the response to get a single word.
        intent = response.text.strip().upper()
        if intent not in ["GENERATE", "MODIFY"]:
            # Fallback heuristic if the model gives an unexpected response
            return "GENERATE"
        return intent
    except Exception as e:
        print(f"An error occurred during intent classification: {e}", file=sys.stderr)
        return "GENERATE" # Default to GENERATE on error

def parse_markdown_to_dict(markdown_text: str) -> dict:
    """
    Parses the full markdown test case into a dictionary for easier handling in a web UI.
    """
    test_case_dict = {}
    lines = markdown_text.split('\n')
    
    # This is a simplified parser. A more robust solution might use regex.
    for line in lines:
        if line.startswith('- **Test Case ID:**'): test_case_dict['test_case_id'] = line.split(':', 1)[1].strip()
        elif line.startswith('- **Title:**'): test_case_dict['title'] = line.split(':', 1)[1].strip()
        elif line.startswith('- **Objective:**'): test_case_dict['objective'] = line.split(':', 1)[1].strip()
        elif line.startswith('- **Scenario:**'): test_case_dict['scenario'] = line.split(':', 1)[1].strip()
        elif line.startswith('- **Prerequisites:**'): test_case_dict['prerequisites'] = line.split(':', 1)[1].strip()

    test_case_dict['test_steps'] = []
    in_table = False
    for line in lines:
        line = line.strip()
        if not line:
            if in_table: break
            continue
        if "Actions" in line and "Expected Result" in line and line.startswith('|'):
            in_table = True
            continue
        if in_table and line.startswith('|--'):
            continue
        if in_table and line.startswith('|'):
            parts = [part.strip() for part in line.split('|')]
            if len(parts) >= 4:
                test_case_dict['test_steps'].append({'Actions': parts[1], 'Expected Result': parts[2], 'Actual Result': parts[3]})
    return test_case_dict

def _read_pdf_content(filepath: str) -> str:
    """Reads text content from a PDF file."""
    try:
        import pypdf
    except ImportError:
        print("\nWarning: `pypdf` is required to read PDF files.", file=sys.stderr)
        print("Please install it using: pip install pypdf", file=sys.stderr)
        return ""
    
    text = []
    try:
        with open(filepath, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        print(f"Warning: Could not read PDF file '{os.path.basename(filepath)}': {e}", file=sys.stderr)
        return ""

def _read_docx_content(filepath: str) -> str:
    """Reads text content from a DOCX file."""
    try:
        import docx
    except ImportError:
        print("\nWarning: `python-docx` is required to read DOCX files.", file=sys.stderr)
        print("Please install it using: pip install python-docx", file=sys.stderr)
        return ""
    
    text = []
    try:
        document = docx.Document(filepath)
        for para in document.paragraphs:
            text.append(para.text)
        return "\n".join(text)
    except Exception as e:
        print(f"Warning: Could not read DOCX file '{os.path.basename(filepath)}': {e}", file=sys.stderr)
        return ""

def _read_txt_content(filepath: str) -> str:
    """Reads text content from a TXT file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not read TXT file '{os.path.basename(filepath)}': {e}", file=sys.stderr)
        return ""

def _chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Splits text into smaller, overlapping chunks."""
    if not text:
        return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def update_vector_index(source_dir: str, index_dir: str, api_key: str):
    """
    Implements the 'Update' step of RAG. It reads documents, chunks them,
    creates vector embeddings, and saves them to an index for fast retrieval.
    """
    print(f"Reading documents from: {source_dir}")
    all_chunks = []
    for filename in sorted(os.listdir(source_dir)):
        filepath = os.path.join(source_dir, filename)
        if not os.path.isfile(filepath):
            continue

        content = ""
        if filename.lower().endswith('.pdf'):
            content = _read_pdf_content(filepath)
        elif filename.lower().endswith('.docx'):
            content = _read_docx_content(filepath)
        elif filename.lower().endswith('.txt'):
            content = _read_txt_content(filepath)

        if content:
            print(f"  - Chunking and processing '{filename}'...")
            chunks = _chunk_text(content)
            # Store where each chunk came from for context
            for chunk in chunks:
                all_chunks.append({"source": filename, "text": chunk})

    if not all_chunks:
        print("Warning: No text could be extracted from documents. Index not updated.")
        return

    print(f"Generated {len(all_chunks)} text chunks. Now creating embeddings...")
    
    # Create embeddings for all chunks
    try:
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        # The 'models/embedding-001' is a powerful model for this task.
        result = genai.embed_content(model='models/embedding-001',
                                     content=chunk_texts,
                                     task_type="RETRIEVAL_DOCUMENT")
        embeddings = np.array(result['embedding'])
        print(f"Successfully created {embeddings.shape[0]} vector embeddings.")

        # Save the chunks and their embeddings
        with open(os.path.join(index_dir, 'chunks.json'), 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f)
        np.save(os.path.join(index_dir, 'embeddings.npy'), embeddings)
        print(f"Vector index saved successfully to '{index_dir}'")

    except Exception as e:
        print(f"Error creating embeddings or saving index: {e}", file=sys.stderr)

def retrieve_relevant_context(scenario: str, index_dir: str, api_key: str, top_k: int = 3) -> str:
    """
    Implements the 'Retrieve' step of RAG. It takes a user scenario, finds the
    most relevant text chunks from the vector index, and returns them.
    """
    chunks_path = os.path.join(index_dir, 'chunks.json')
    embeddings_path = os.path.join(index_dir, 'embeddings.npy')

    if not (os.path.exists(chunks_path) and os.path.exists(embeddings_path)):
        print("Warning: Vector index not found. Run with --update-index first. Continuing without context.")
        return ""

    print("Retrieving relevant context from vector index...")
    try:
        # Load the pre-computed index
        with open(chunks_path, 'r', encoding='utf-8') as f:
            all_chunks = json.load(f)
        doc_embeddings = np.load(embeddings_path)

        # Create an embedding for the user's scenario (the query)
        query_embedding = genai.embed_content(model='models/embedding-001',
                                              content=scenario,
                                              task_type="RETRIEVAL_QUERY")['embedding']

        # Find the most similar documents using cosine similarity
        query_embedding = np.array(query_embedding)
        similarities = np.dot(doc_embeddings, query_embedding) / (np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding))

        # Get the indices of the top_k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # --- Add a relevance threshold to filter out irrelevant results ---
        # This prevents documents that are only vaguely related from "polluting" the context.
        relevance_threshold = 0.65 # Tuned to be slightly more permissive.
        relevant_chunks = []
        for i in top_indices:
            if similarities[i] >= relevance_threshold:
                print(f"  - Found relevant chunk from '{all_chunks[i]['source']}' (Similarity: {similarities[i]:.2f})")
                relevant_chunks.append(f"--- Context from {all_chunks[i]['source']} ---\n{all_chunks[i]['text']}")
            else:
                # Since the list is sorted by similarity, a score below the threshold means the rest are also irrelevant.
                print(f"  - Discarding chunk from '{all_chunks[i]['source']}' (Similarity: {similarities[i]:.2f} < {relevance_threshold})")

        if not relevant_chunks:
            print("  - No documents met the relevance threshold. Proceeding without custom context.")
            return ""

        return "\n\n".join(relevant_chunks)

    except Exception as e:
        print(f"Error retrieving context from vector index: {e}", file=sys.stderr)
        return ""

def _create_example_section(template_path: str | None = None, template_stream=None) -> str:
    """Creates the full example section for the prompt, either from a template file or a hardcoded default."""
    
    # Prioritize the stream if it exists (from web UI), otherwise use the path (from CLI).
    template_source = template_stream
    if not template_source and template_path and os.path.exists(template_path):
        template_source = template_path

    if template_source:
        print(f"Attempting to use template file...")
        try:
            import pandas as pd
            # Assuming the template's test steps are on the first sheet
            df = pd.read_excel(template_source, engine='openpyxl')
            template_md = df.to_markdown(index=False)
            # Return a prompt section that specifically guides the AI to use the template for the steps
            return f"""
**Template Guidance:**
When creating the 'Test Steps' table, you MUST follow the exact column structure, format, and wording style demonstrated in this example from the user-provided template:
---
{template_md}
---
"""
        except Exception as e:
            print(f"Warning: Could not process template file: {e}. Using default example.", file=sys.stderr)
            # Fall through to default if template processing fails

    # Default hardcoded example if no template is provided or if it fails to load
    return """
**Guidance for Test Step Columns:**
- **Actions:** Step-by-step user actions to be performed in the system (e.g., navigation, form entry, clicks). Clearly list the steps in sequence. Use correct Maximo navigation terms (e.g., Go To Applications → Work Order Tracking).
- **Expected Result:** The system behavior or output that should occur if the action is executed correctly. Be precise (e.g., "Status changes to APPR", "Field becomes read-only").
- **Actual Result:** What would be observed during a successful test execution. This should confirm the expected result was met, written in the past tense.

**Example Output Format:**

# Test Case: Verify Purchase Requisition Creation Restriction
- **Test Case ID:** TC-MAX-001
- **Title:** Verify Purchase Requisition Creation Restriction
- **Objective:** To verify that the system prevents the creation of a Purchase Requisition when the HCC question is set to 'Yes', directing the user to SAP.
- **Prerequisites:** User is logged into Maximo with `BUYER` role and permissions for Purchase Requisitions.
- **Test Steps:**
    | Actions                                                                    | Expected Result                                                              | Actual Result                                                                 |
    |----------------------------------------------------------------------------|------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
    | 1. Navigate to Applications → Purchasing → Purchase Requisitions.          | Purchase Requisitions application opens successfully.                        | Purchase Requisitions application opened successfully.                        |
    | 2. Change HCC question from 'No' to 'Yes'.                                 | Error message appears: 'Purchase Requisition cannot be created in Maximo. Please create it in SAP.' | Error message was immediately triggered after selecting 'Yes'.                |
"""

def main():
    """
    Main function to parse arguments, generate the test case, and save it.
    """
    parser = argparse.ArgumentParser(
        description="Generate an IBM Maximo test case in Markdown format using the Gemini API.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "scenario", # This is now optional
        nargs='?',
        default=None,
        type=str,
        help="The test scenario to be converted into a test case. Enclose in quotes."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="maximo_test_case.md",
        help="The name of the output Markdown file. (default: maximo_test_case.md)"
    )
    parser.add_argument(
        "--update-index",
        action='store_true', # Makes it a flag, e.g., --update-kb
        help="Update the vector index from the 'Learning_Maximo' folder. Run this after changing documents."
    )
    
    args = parser.parse_args()

    # Define project directories for better organization
    project_dir = r'C:\Users\2166337\Test_Case'
    learning_dir = os.path.join(project_dir, 'Learning_Maximo')
    index_dir = os.path.join(project_dir, 'Maximo_VectorIndex')
    output_dir = project_dir # Save output files in the main project folder

    os.makedirs(learning_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    # --- API Key for Command-Line Mode ---
    cli_api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyBCIP9nSgxdJmMaBwITcbuFZ81dC9bzJLQ")
    if "AIzaSyBCIP9nSgxdJmMaBwITcbuFZ81dC9bzJLQ" in cli_api_key:
        print("WARNING: Using hardcoded fallback API key for command-line execution.")

    # Configure genai once for command-line use
    try:
        genai.configure(api_key=cli_api_key)
        print("✅ Google Generative AI configured for CLI.")
    except Exception as e:
        print(f"❌ ERROR: Failed to configure Google Generative AI for CLI: {e}")
        sys.exit(1)

    # --- Mode 1: Update the Knowledge Base ---
    if args.update_index:
        print("--- Updating Vector Index ---")
        update_vector_index(learning_dir, index_dir, cli_api_key)
        print("--- Vector Index update complete. ---")
        sys.exit(0) # Exit successfully after updating

    # --- Mode 2: Generate a Test Case (Normal Operation) ---
    if not args.scenario:
        parser.error("The 'scenario' argument is required when not using --update-kb.")

    # This main block is for command-line use only. The web app calls functions directly.
    # For simplicity, we'll use a hardcoded example section for CLI mode.
    example_section = _create_example_section(None)
    api_keys = {"google": cli_api_key} # CLI mode only supports Google for now
    model_name = 'gemini-1.5-flash-latest'

    custom_context = ""
    custom_context = retrieve_relevant_context(args.scenario, index_dir, api_keys['google'])
    if custom_context:
        print("Successfully retrieved relevant context from the vector index.")
    
    test_case_markdown = generate_maximo_test_case(args.scenario, api_keys, custom_context, example_section, model_name)
    
    # Construct full paths for the output files
    output_md_path = os.path.join(output_dir, args.output)
    
    try:
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write(test_case_markdown)
        print(f"\nSuccessfully generated and saved test case to '{os.path.abspath(output_md_path)}'")
    except IOError as e:
        print(f"Error writing to file '{output_md_path}': {e}", file=sys.stderr)
        # We can still try to create the excel file, so we don't exit here.

    # Save the extracted test steps to a separate Excel file.
    base_name, _ = os.path.splitext(args.output)
    excel_filename = f"{base_name}_steps.xlsx"
    output_xlsx_path = os.path.join(output_dir, excel_filename)
    save_steps_to_excel(test_case_markdown, output_xlsx_path)

if __name__ == "__main__":
    main()
