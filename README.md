## Cómo correr (end-to-end)
Desde la raíz del repo:

docker compose down -v --remove-orphans
docker compose build --no-cache
docker compose up -d db
docker compose run --rm app
