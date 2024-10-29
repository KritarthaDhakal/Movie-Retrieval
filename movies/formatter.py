import re


def format_docname(doc):

    name_without_extension = doc.replace('.txt', '')
    split_name = re.sub(r'(\d+|[a-z])([A-Z])', r'\1 \2', name_without_extension)
    split_name = re.sub(r'(Part)(\d+)', r'\1 \2', split_name)
    formatted_name = split_name.title()
    return formatted_name


def extract_content(document_content_dict, documents):

    extracted_dict = {}
    for doc in documents:
        if doc in document_content_dict:
            words = document_content_dict[doc].split()
            formatted_name = format_docname(doc)
            extracted_dict[formatted_name] = ' '.join(words[:27])
    return extracted_dict