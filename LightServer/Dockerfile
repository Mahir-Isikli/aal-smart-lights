FROM node:14-bullseye-slim AS build

WORKDIR /usr/src/app

COPY . .

RUN npm install \
    && npm run build

FROM node:14-bullseye-slim AS run

ENV NODE_ENV=production
WORKDIR /usr/src/app

COPY package.json ./
COPY --from=build /usr/src/app/package-lock.json ./
RUN npm ci

COPY --from=build /usr/src/app/dist ./

ENV PORT=8080
EXPOSE 8080
CMD ["node", "./index.js"]
