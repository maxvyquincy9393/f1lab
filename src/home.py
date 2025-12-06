import streamlit as st
import pandas as pd
import fastf1
from datetime import timedelta

def render_home_tab():
    """Tab 0: Home - Dynamic Content based on Event Status."""
    st.header("F1 Dashboard Home")
    
    # Current time (UTC)
    now = pd.Timestamp.now(tz='UTC')
    
    try:
        schedule = fastf1.get_event_schedule(2025)
        
        # Ensure timezone awareness
        if schedule['EventDate'].dt.tz is None:
             schedule['EventDate'] = schedule['EventDate'].dt.tz_localize('UTC')
        
        # Determine current/next event
        # Logic: Find first event where last session hasn't finished yet OR closest future event
        # For simplicity, let's look for the next upcoming event or active one
        
        # Filter for events where the event date (Sunday race) is in the future or just passed (within last 3 days)
        upcoming_events = schedule[schedule['EventDate'] >= (now - timedelta(days=3))].sort_values('EventDate')
        
        if not upcoming_events.empty:
            target_event = upcoming_events.iloc[0]
            event_name = target_event['EventName']
            round_num = target_event['RoundNumber']
            location = target_event['Location']
            event_date = target_event['EventDate']
            
            # Check if Live (approximate: race start <= now <= race end + buffer)
            # Better: Check individual sessions
            is_live = False
            live_session_name = ""
            
            sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
            for s in sessions:
                if f'{s}Date' in target_event and pd.notna(target_event[f'{s}Date']):
                    s_date = target_event[f'{s}Date']
                    if isinstance(s_date, pd.Timestamp) and s_date.tz is None:
                        s_date = s_date.tz_localize('UTC')
                    
                    s_end = s_date + timedelta(hours=2.5) # Generous buffer
                    
                    if s_date <= now <= s_end:
                        is_live = True
                        live_session_name = target_event[s]
                        break
            
            if is_live:
                # --- LIVE MODE ---
                st.success(f"(LIVE) {event_name} - {live_session_name}")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"### {event_name}")
                    st.markdown(f"**Round {round_num}** | {location}")
                    
                    st.markdown("---")
                    st.markdown("##### Quick Actions")
                    if st.button("Go to Live Timing >", type="primary"):
                        st.session_state.selected_tab = "Live Timing" # We need to handle this state change in main
                        st.rerun()
                        
                with col2:
                    # Maybe show weather or simple status if possible without heavy loading
                    st.info("Session is currently active. Switch to Live Timing tab for real-time data.")
                    
            else:
                # --- UPCOMING / RECAP MODE ---
                days_until = (event_date - now).days
                
                if days_until < 0:
                    # Just finished
                    st.info(f"Just Concluded: {event_name}")
                    st.markdown("Check out the analysis tabs for full results.")
                else:
                    # Upcoming
                    st.markdown(f"### Next Up: {event_name}")
                    st.markdown(f"**{location}** | {event_date.strftime('%d %b %Y')}")
                    
                    if days_until <= 7:
                        st.warning(f"IT'S RACE WEEK! ({days_until} days to go)")
                    else:
                        st.markdown(f"*{days_until} days until lights out*")
            
            st.divider()
            st.subheader("2025 Season Schedule")
            
            # Simplified schedule table
            display_schedule = upcoming_events[['RoundNumber', 'EventDate', 'EventName', 'Location']].head(5).copy()
            display_schedule['EventDate'] = display_schedule['EventDate'].dt.strftime('%Y-%m-%d')
            display_schedule.columns = ['Round', 'Date', 'Event', 'Location']
            st.table(display_schedule)
            
        else:
            st.info("No upcoming events found in schedule.")
            
    except Exception as e:
        st.error(f"Could not load home dashboard: {e}")
