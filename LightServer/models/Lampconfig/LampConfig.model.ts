import { Schema, model } from 'mongoose';
import LampConfig from "./LampConfig.interface";

const schema = new Schema<LampConfig>({
    id: {type: String, required: true, unique: true},
    lampId: String,
    turnedOn: Boolean,
    color: String,
    intensity: Number
})

export default model("Lampconfig", schema)
