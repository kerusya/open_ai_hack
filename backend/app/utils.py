number_sent = 5

changed_schema_description = 'Противоречит ли информация из изначального документа информации из нового документа? \
Вернуть значение True, если информация из имеющегося документа по смыслу прямо противоречит информации из нового документа, \
False если изначальный и новый документ не имеют противоречий по смыслу.'

new_text_schema_description = 'Если на предыдущий вопрос возвращается True - надо переписать текст изначального документа \
таким образом, чтобы он соответстовал новому документу по смыслу и вернуть эту версию, оставив те части текста, которые ничему не \
противоречат. Если на предыдущий вопрос возвращается False - вернуть оригинальный текст документа.'

explanation_schema_description = 'Обоснуй принятые решения и сделай выводы, пожалуйста.'

template_string = '''

У нас есть документ, по которому осуществляется кредитная деятельность банка. Есть второй документ, содержащий новый законный акт.
Мы хотим определить, противоречит ли внутренний документ банка новому законному акту. Для этого необходимо сверить эти два \
документа на предмет несоответствий. Документы могут быть на разные темы и регулировать разные аспекты деятельности банка, такие \
случаи мы не считаем несоответствиями, мы ищем только прямые логические различия в текстах документов.

Изначальный документ: 
{original_document}

Новый документ:
{new_document}

{format_instructions}
'''
