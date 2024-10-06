import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './ChatWidget.css';
import { FaPaperPlane } from 'react-icons/fa'; // Only keep the paper plane icon

const ChatWidget = () => {
    const [messages, setMessages] = useState([]);
    const [userMessage, setUserMessage] = useState('');
    const chatBodyRef = useRef(null);

    useEffect(() => {
        if (chatBodyRef.current) {
            chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
        }
    }, [messages]);

    const handleSendMessage = async () => {
        if (userMessage.trim() === '') return;

        const newMessage = { role: 'user', content: userMessage };
        setMessages((prevMessages) => [...prevMessages, newMessage]);
        setUserMessage('');

        try {
            const response = await axios.post('http://localhost:5000/recipe', {
                ingredients: userMessage,
            });

            const botMessage = response.data.response
                ? { role: 'assistant', content: response.data.response }
                : { role: 'assistant', content: 'Sorry, I did not understand your request.' };
            setMessages((prevMessages) => [...prevMessages, botMessage]);
        } catch (error) {
            const errorMessage = { role: 'assistant', content: 'An error occurred while processing your request.' };
            setMessages((prevMessages) => [...prevMessages, errorMessage]);
        }
    };

    const getMessage = (msg, index) => {
        var response = msg.content.split('\n');
        return (
            <div key={index} className={`message ${msg.role}`}>
                {response.map((responseItem, i) => (
                    <p key={i}>{responseItem}</p>
                ))}
            </div>
        );
    };

    return (
        <div className="chat-widget">
            <div className="chat-header">National Foods</div>
            <div className="chat-body" ref={chatBodyRef}>
                <div className="chat-body-background">
                    
                </div>
                {messages.map(getMessage)}
            </div>
            <div className="chat-footer">
                <input
                    type="text"
                    placeholder="Type your message..."
                    value={userMessage}
                    onChange={(e) => setUserMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                />
                <button onClick={handleSendMessage}>
                    <FaPaperPlane />
                </button>
            </div>
        </div>
    );
};

export default ChatWidget;