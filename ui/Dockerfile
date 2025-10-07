# ---- build stage ----
FROM node:18-alpine AS build
WORKDIR /app

# 1. Paket dosyaları
COPY package*.json ./

# 2. Config dosyaları (bunlar eksikti)
COPY tsconfig.json vite.config.js tailwind.config.js postcss.config.js ./

# 3. Uygulama kaynakları
COPY index.html ./index.html
COPY src ./src

# 4. Kurulum + build
RUN npm ci || npm i
RUN npm run build

# ---- run stage ----
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
