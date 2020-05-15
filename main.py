import os
import config
from discord.ext import commands

bot = commands.Bot(command_prefix = '$')
bot.remove_command('help')

@bot.event
async def on_ready():

    print('-'*34)
    print('Logged in as: ', bot.user.name)
    print('Client ID:    ', bot.user.id)
    print('Local time:   ', config.SERVER_TIME)
    print('-'*34)

@bot.event
async def on_message(context):
    await bot.process_commands(context)

def load_extensions():
    # Loads all of the extensions. Note: check iDM if I branch out to multiple folders
    exclusion_list = []
    for filename in os.listdir('./ocr'):
        cog = filename[:-3]
        if filename.endswith('.py') and cog not in exclusion_list:
            try:
                bot.load_extension(f'ocr.{cog}')
                print(f'Loaded extension: {cog}')
            except Exception as err:
                exc = f'{type(err).__name__}: {err}'
                print(f'Failed to load extension {cog}\n{exc}')

def log_in():
    load_extensions()
    print('Attempting to log in...')
    try:
        bot.run(config.DISCORD_TOKEN)
    except Exception as error:
        print('Discord: Unsuccessful login. Error: ', error)
        quit()

if __name__ == '__main__':
    log_in()