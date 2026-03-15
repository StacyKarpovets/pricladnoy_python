# URL Shortener Service

API-сервис для сокращения ссылок.

Документация API: [https://pricladnoy-python.onrender.com/docs](https://pricladnoy-python.onrender.com/docs)

### Основной функционал
- **Создание короткой ссылки** — `POST /links/shorten`
- **Редирект по короткой ссылке** — `GET /{short_code}`
- **Удаление ссылки** — `DELETE /links/{short_code}`
- **Обновление ссылки** — `PUT /links/{short_code}`
- **Статистика по ссылке** — `GET /links/{short_code}/stats`
- **Поиск по оригинальному URL** — `GET /links/search?original_url={url}`
- **Время жизни ссылки** — параметр `expires_at` при создании

### Дополнительные функции
- **Регистрация и JWT-аутентификация**
- **Redis-кэширование популярных ссылок**
- **Автоматическая очистка истекших ссылок**
- **Создание ссылок без регистрации**

## Стэк
- **FastAPI** — веб-фреймворк
- **PostgreSQL** — основная база данных
- **Redis** — кэширование и счетчики переходов
- **SQLAlchemy** — ORM
- **JWT** — аутентификация
- **Render** — хостинг и деплой

## Пример запроса

### Регистрация пользователя

curl -X POST https://pricladnoy-python.onrender.com/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "email": "test@example.com", "password": "password123"}'
### Получение токена

curl -X POST https://pricladnoy-python.onrender.com/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=password123"
###  Создание короткой ссылки

curl -X POST https://pricladnoy-python.onrender.com/links/shorten \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"original_url": "https://www.google.com"}'
### Получение статистики

curl https://pricladnoy-python.onrender.com/links/DIJ2qS/stats
