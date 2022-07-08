import Lamp from "../models/Lamp/Lamp.interface"
import LampModel from "../models/Lamp/lamp.model"
import LampConfigModel from "../models/Lampconfig/LampConfig.model"
import LampConfig from "../models/Lampconfig/LampConfig.interface";
import express, { Request, Response } from 'express'

const http = require('http')

export default class LampController {
    // kennt alle Lamps (+ add, delete, modify)

    constructor() {
        /*
        const config1: Lampconfig = {
            id: "lukeOn", lampId: "LukeRoberts", color: "white", intensity: 100, turnedOn: true
        }

        const config2: Lampconfig = {
            id: "stripOn", lampId: "LightStrip", color: "blue", intensity: 80, turnedOn: true
        }

         */

    }

    create = (req: Request, res: Response) => {
        // Validate request
        if(!req.body) {
            return res.status(400).send({
                message: "body can not be empty"
            });
        }

        // Create a config
        const config = new LampConfigModel({
            id: req.body.id,
            lampId: req.body.lampId,
            turnedOn: req.body.turnedOn,
            color: req.body.color,
            intensity: req.body.intensity
        })

        // Save config in the database
        config.save()
            .then(data => {
                res.send(data);
            }).catch(err => {
            res.status(500).send({
                message: err.message || "Some error occurred while creating the Config."
            });
        });
    }

    find = (req: Request, res: Response) => {
        if (req.query.id) this.findOne(req, res)
        else this.findAll(req, res)
    }

    findAll = (req: Request, res: Response) => {
        LampConfigModel.find()
            .then(configs => {
                res.send(configs);
            }).catch(err => {
            res.status(500).send({
                message: err.message || "Some error occurred while retrieving configs."
            });
        });
    }

    findOne = (req: Request, res: Response) => {
        LampConfigModel.findOne({id: req.query.id})
            .then(config => {
                if(!config) {
                    return res.status(404).send({
                        message: "Config not found with id " + req.query.id
                    });
                }
                res.send(config);
            }).catch(err => {
            if(err.kind === 'ObjectId') {
                return res.status(404).send({
                    message: "Config not found with id " + req.query.id
                });
            }
            return res.status(500).send({
                message: "Error retrieving config with id " + req.query.id
            });
        });
    }

    update = (req: Request, res: Response) => {
        // Validate Request
        if(!req.body.id) {
            return res.status(400).send({
                message: "Config ID can not be empty"
            });
        }

        // Find note and update it with the request body
        LampConfigModel.findOneAndUpdate({id: req.query.id}, {
            id: req.body.id,
            lampId: req.body.lampId || undefined,
            turnedOn: req.body.turnedOn || undefined,
            color: req.body.color || undefined,
            intensity: req.body.intensity || undefined
        }, {new: true})
            .then(config => {
                if(!config) {
                    return res.status(404).send({
                        message: "Config not found with id " + req.query.id
                    });
                }
                res.send(config);
            }).catch(err => {
            if(err.kind === 'ObjectId') {
                return res.status(404).send({
                    message: "Config not found with id " + req.query.id
                });
            }
            return res.status(500).send({
                message: "Error updating config with id " + req.query.id
            });
        });
    }

    delete = (req: Request, res: Response) => {
        LampConfigModel.findOneAndRemove({id: req.query.id})
            .then(config => {
                if(!config) {
                    return res.status(404).send({
                        message: "Config not found with id " + req.query.id
                    });
                }
                res.send({message: "Config deleted successfully!"});
            }).catch(err => {
            if(err.kind === 'ObjectId' || err.name === 'NotFound') {
                return res.status(404).send({
                    message: "Config not found with id " + req.query.id
                });
            }
            return res.status(500).send({
                message: "Could not delete config with id " + req.query.id
            });
        });
    }

    /**
     * Parses a LampConfig with type 'wled' into a String that can be sent via HTTP GET request
     * @param config LampConfig of type 'wled'
     * @param lamp Lamp that the LampConfig is meant for
     */
    configToWledRequest(config: LampConfig, lamp: Lamp): string {
        console.log("Building WLED Req!")
        let result = "" // will be returned
        const _config: LampConfig = {id: config.id, // copy config so we dont mess with it
            lampId: config.lampId,
            color: config.color,
            turnedOn: config.turnedOn,
            intensity: config.intensity}
        _config.color = (_config.color?.length != 6) ? "FFFFFF" : _config.color // make sure color is well defined

        // translate config to WLED HTTP API format
        // from: https://kno.wled.ge/interfaces/http-api/

        // http://1.2.3.4&A=255&T=1&R=255&G=255&B=255
        const wledConfig: any = {
            A: Math.floor(_config.intensity / 100 * 255), // brightness
            T: _config.turnedOn ? 1 : 0, // turned on/off
            R: parseInt(_config.color.slice(0, 2), 16), // get red value
            G: parseInt(_config.color.slice(2, 4), 16), // get blue value
            B: parseInt(_config.color.slice(4, 6), 16)  // get green value
        }

        // build HTTP API string from Lamp IP and config
        result = `http://${lamp.ip}/win`
        for (let prop in wledConfig) {
            result += `&${prop}=${wledConfig[prop]}`
        }

        return result
    }
    /**
     * Führt eine LampConfig aus,
     * indem mit dem richtigen Protokoll eine Anfrage an das Gerät geschickt wird
     */
    executeLampConfig(configID: String): Promise<undefined> {
        return new Promise((resolve, reject) => {
            LampConfigModel.findOne({id: configID})
                .then(async config => {
                    console.log("[Lamp Controller] Executing config: ", config)

                    // make sure config is defined
                    if (!config) {
                        console.log("err! no config: " + configID)
                        reject("[Lamp Controller] Error! Got empty config for: " + configID)
                        return
                    }

                    // get Lamp from our db
                    const lamp = await LampModel.findOne({id: config.lampId})
                    if (!lamp) {
                        console.log("err! no lamp: " + config.lampId)
                        reject(`[Lamp Controller] Error executing config: ${config} Could not find lampID: ${config.lampId}`)
                        return
                    }

                    // parse config depending on type of lamp
                    // then send parsed request off to the lamp
                    if (lamp.type == "wled") {
                        const parsedConfig = this.configToWledRequest(config, lamp)
                        console.log("[Lamp Controller] Sending request: ", parsedConfig)
                        http.get(parsedConfig)
                    }
                    else console.log("unknown protocol, config: " + config)

                    resolve(undefined)
                })
                .catch((err) => {
                    console.log("err! running config: " + configID + "\n"+ err)
                    reject("[Lamp Controller] could not find config: " + configID)
                })
        })
    }
}
