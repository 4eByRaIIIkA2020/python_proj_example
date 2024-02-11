import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from vk_api.upload import VkUpload
import random
import requests
from io import BytesIO


# Токен группы
token = '367f8649a5141b4ba1430a6151b81192a86ddb74a9c237e2fc6ad0a208727fff26570d8ee3ec67708be0d'
vk_session = vk_api.VkApi(token = token)
vk = vk_session.get_api()
longpoll_group = VkBotLongPoll(vk_session, 192205974)  # ID группы

chat_id = 1  # Куда отправлять репост(id беседы)

upload = VkUpload(vk)


def send_photo(vk, chat_id, owner_id, photo_id, access_key):
    attachment = f'photo{owner_id}_{photo_id}_{access_key}'
    vk.messages.send(
        random_id = 0,
        message = "Ура, новая новость в паблике: ",
        chat_id = chat_id,
        attachment=attachment)#отправка сообщений в беседу

        
def upload_photo(upload, url):#загрузка картинки по url
    img = requests.get(url).content
    f = BytesIO(img)
    response = upload.photo_messages(f)[0]
    owner_id = response['owner_id']
    photo_id = response['id']
    access_key = response['access_key']
    return owner_id, photo_id, access_key



for event in longpoll_group.listen():
    if event.type == VkBotEventType.WALL_POST_NEW:

        for item in event.object['attachments']:
            if item['type'] == 'photo':#определение типа вложений(фото)
                url = item['photo']['sizes'][-1]['url']#ссылка на картинку
                send_photo(vk, chat_id, *upload_photo(upload, url))#отправляем сообщение + делаем загрузку картинки по url

