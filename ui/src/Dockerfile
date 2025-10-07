# Build
FROM node:18-alpine AS build
WORKDIR /app
COPY package.json vite.config.* tsconfig.* ./
RUN npm ci || npm i
COPY index.html ./index.html
COPY src ./src
RUN npm run build

# Serve
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# (opsiyonel) Nginx'e /api proxy eklerseniz burada conf kopyalanır
