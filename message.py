import stomp
import urllib  
import time

def sendMessageToSmartBody(cmd):
        try:
                conn = stomp.Connection()#comment this line
                #conn.start()
                conn.connect('admin', 'password', wait=True)
                body = cmd
                message = urllib.quote_plus(body)
                hdrs = {}
                hdrs['ELVISH_SCOPE'] = 'DEFAULT_SCOPE'
                hdrs['MESSAGE_PREFIX'] = 'sbm'
                conn.auto_content_length= False
                conn.send(body=message, headers=hdrs, destination='/topic/DEFAULT_SCOPE')
                #time.sleep(0.040)
                #conn.disconnect()#comment this line
        except Exception as e:
                print e.message

#sendMessageToSmartBody("sb scene.getDiphoneManager().setPhonemesRealtime('foo', 'oo')")
#sendMessageToSmartBody("sb print x")
