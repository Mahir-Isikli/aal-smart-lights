import { Schema, model } from 'mongoose';
import Lamp from "./Lamp.interface";

const schema = new Schema<Lamp>({
    id: {type: String, required: true, unique: true},
    type: {type: String, required: true},
    ip: String,
    hasColor: Boolean,
    hasIntensity: Boolean,
    category: Number,
    location: Number
})

export default model("Lamp", schema)

