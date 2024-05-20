# Moody Lyrics

Song recommendation website.

## High-Level Functional Features

- Associates a song with a mood based on the lyrics.
- Lists out similar songs.

## Eventual Non-Functional Features

- Scale - 1 million reqs/second
- Availability - Limited downtime

### High Level Architecture

- A simple web app that takes in song name and artist.
- A backend server that -
  - Fetches the lyrics - possibly on genius.com
  - Passes the lyrics into the model to assign a mood.
  - Uses this mood to fetch similar songs.
- Backend served and scaled by TFX?

### Tech Stack

- Front-end - React
- Rest of the stuff in Python.
- Database - TDB
