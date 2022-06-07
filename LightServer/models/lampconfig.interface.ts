import { Schema, model } from 'mongoose';

interface Lampconfig {
    id: String,
    lampId: String,
    turnedOn: boolean,
    color: String,
    intensity: Number
}

const schema = new Schema<Lampconfig>({
    id: {type: String, required: true, unique: true},
    lampId: String,
    turnedOn: Boolean,
    color: String,
    intensity: Boolean
})

export default model("Lampconfig", schema)
