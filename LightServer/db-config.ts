const mongoose = require('mongoose')
const url = process.env.DB_CONNECTION_STRING || 'mongodb://127.0.0.1:27017/LightServer'

mongoose.connect(url, { useNewUrlParser: true })
const db = mongoose.connection
db.once('open', () => {
    console.log('Database connected:', url)

    /*
    // DEBUG: uncomment to print all collections in db
    mongoose.connection.db.listCollections().toArray(function (err: any, names: any) {
        console.log(names); // [{ name: 'dbname.myCollection' }]
        module.exports.Collection = names;
    });
     */
})

db.on('error', (err: any) => {
    console.error('connection error:', err)
})

export default db

