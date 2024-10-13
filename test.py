prompt_suffix ="You are an AI assistant who specializes entity types. Your task is as follows: according to the sentence, predict the type of entity mention in the sentence. " \
                   "If the predicted type belongs to the known types supported by the system, return the corresponding known type, otherwise return 'unknown type'. The supported known types include: "
fine_grained_keys=[repr('person'),'location','gpe']
prompt_suffix += " ".join(fine_grained_keys)
print(prompt_suffix)