import express, { Express } from 'express';
import dotenv from 'dotenv';
import swaggerUi from 'swagger-ui-express'
import swaggerJsdoc from 'swagger-jsdoc'

import lightRoutes from './src/routes/lights/toggle.route'
import eventRoutes from "./src/routes/events/trigger.route"
dotenv.config();
const app: Express = express();
const cors = require('cors')

/**
 * Creates an Express server, then add all routes and configuration
 *
 * Returns the configured express instance
 */

app.use(cors({ credentials: true, origin: true }))

app.use('/lights', lightRoutes);
app.use('/events', eventRoutes);

const options = {
    definition: {
        openapi: '3.0.0',
        info: {
            title: 'Smart lights',
            version: '1.0.0',
        },
    },
    apis: [
        './routes/lights/*.ts']
}

const swaggerSpec = swaggerJsdoc(options)
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec))

app.use(express.static("public"));

export default app
