---
layout: post
title: CackalackyCon 2024 Crypto Challenge Solutions
date: 2024-09-24 04:04:06 -0400
tags: 
- crypto
- osint
- CackalackyCon
- ctf
---

Hi! I'm the person who created the CackalackyCon 2024 Crypto Challenge. I wanted to provide the solutions to the challenge for anyone who was curious.

# Step 1

### Message
`N7 is hunting a spacecriminal.com! The secret code is this longest serving turian's name. Sometimes the most significant is the least significant CRC32 PKCS #7 AES-256-CBC`
### Encryption
Atbash the message 
`M7 rh sfmgrmt z hkzxvxirnrmzo.xln! Gsv hvxivg xlwv rh gsrh olmtvhg hviermt gfirzm'h mznv. Hlnvgrnvh gsv nlhg hrtmrurxzmg rh gsv ovzhg hrtmrurxzmg XIX32 KPXH #7 ZVH-256-XYX`
encoded in a qr code
### Ciphertext
<img src="/assets/cackalacky/2024/step1.png" width="800px">

# Step 2
Github pages storing an image. Image contains an animalese mp3 file.
Animalese pointing at "defcon dot social slash at kiwi"
[defcon.social/@kiwi](https://defcon.social/@kiwi "https://defcon.social/@kiwi")
Players must translate the animalese to english. 
### Message
`defcon dot social slash at kiwi`
### Encryption
[https://github.com/equalo-official/animalese-generator](https://github.com/equalo-official/animalese-generator "https://github.com/equalo-official/animalese-generator")

wav file is encoded into the image using https://github.com/7thSamurai/steganography 
password is saren


The program operates by first randomly generating a _128-bit Password Salt_ and a _128-bit AES Initialization Vector_ by reading binary data from **/dev/urandom**. It then uses that _Password Salt_ as a parameter in generating an encryption key, by using **PBKDF2-HMAC-SHA-256** on a user inputted string. A **CRC32** hash of the file to embed is then calculated, and stored in the header to act as a checksum for the validity of the data. It then pads the binary data of the file to embed using the **PKCS #7** algorithm, followed by actually encrypting both the header and the padded data, with **AES-256** in **CBC Mode**, using the previously generated _Initialization Vector_. Now the data is actually encoded inside the image by first picking a random offset, and then going through each bit of data and storing it inside the actual image pixel data, which it accomplishes by setting the _Least-Significant-Bit_ of each channel byte of each pixel.
### Ciphertext
[Animalese Sound Byte](https://discord.com/channels/@me/1068716779561566300/1240875726547320833)
# Step 3
I repost fekie's red panda with the phrase "These animals belong in siberia"

fekie responds, we flame each other back and forth 
I drop some like "Even zuko is smarter than you!" with a picture of zuko
- last message, fekie says "This is gonna drop harder than meet the grahams. I'm exposing you. We'll see who belongs in siberia"
	fekie posts an encrypted message that points to a website after the red panda post
### Message
secrets.kiwi.observer
secrets dot kiwi dot observer
кистэлэҥнэр точка киви точка байкоочу
### Encryption
English -> yakut
base64 encrypted
### Ciphertext
Base32 encrypted
`2C5NBOGRQHIYFUMN2C55DDOSUXIL3UMN2GACBUMC2C7NDB6QXLILAIGQXLILRUFS2C4CBUMC2C7NDB6QXLILAIGQWHILBUFZ2C5NBPWQX3IYPUMD`
# Step 4
Website has a DNS record containing an encrypted message.
Can be found using dig tool, or https://toolbox.googleapps.com/apps/dig/#TXT/ 
Point at kiwi.observer
### Message

`ctf room password is zuko`
![[/assets/cackalacky/2024/hylian.png]]
![[/assets/cackalacky/2024/hylian_fuzzy.png]]
### Encryption
Text is converted to hylian (legend of zelda) and hint is given
Hint: `I'm feeling rather dubious about crunching numbers to break this encryption.`
### Ciphertext
![[/assets/cackalacky/2024/DNS.png]]
# Step 5
CTF people tell anyone who says "zuko" to them 

"First one to to DM Kiwi on discord how to cheat at Contra wins. "

First person to DM me and say
up up down down left right left right b a select start
or some variant wins
Trophy:![[/assets/cackalacky/2024/trophy.png]]